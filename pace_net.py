"""
Preprocessing Utilities for PACE-Net
======================================
Wraps fMRIPrep, MNE-Python, and FSL pipelines.
These functions assume preprocessing has already been run
by the respective tools and produce the numpy arrays
expected by ADNIDataset.

Pipeline summary (from paper Section 3.2):
  fMRI  : fMRIPrep v23.2 → Schaefer 200-ROI parcellation
  EEG   : MNE-Python → bandpass → ICA → stage labels (Dreem 2)
  DTI   : FSL TBSS → eddy correction → probtrackx2 tractography
  T1-MRI: FreeSurfer v7.3 → PVSSEG PVS segmentation
"""

import os
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple


# ─── fMRI Preprocessing ───────────────────────────────────────────────────────

def extract_roi_timeseries(fmri_preprocessed: np.ndarray,
                            atlas_labels: np.ndarray,
                            n_rois: int = 200) -> np.ndarray:
    """
    Extract mean BOLD time series per ROI using the Schaefer 200-ROI atlas.

    Args:
        fmri_preprocessed: (X, Y, Z, T) 4D preprocessed fMRI volume
        atlas_labels:      (X, Y, Z)    atlas label volume (Schaefer 200)
        n_rois:            number of ROIs (200 for Schaefer-200)
    Returns:
        roi_ts: (n_rois, T) mean time series per ROI
    """
    T = fmri_preprocessed.shape[-1]
    roi_ts = np.zeros((n_rois, T))
    for roi_id in range(1, n_rois + 1):
        mask = (atlas_labels == roi_id)
        if mask.sum() > 0:
            voxels = fmri_preprocessed[mask]  # (n_voxels, T)
            roi_ts[roi_id - 1] = voxels.mean(0)
    return roi_ts


def normalise_bold(roi_ts: np.ndarray) -> np.ndarray:
    """
    Z-score normalise BOLD time series per ROI.
    """
    mean = roi_ts.mean(axis=1, keepdims=True)
    std  = roi_ts.std(axis=1, keepdims=True)
    return (roi_ts - mean) / (std + 1e-8)


def compute_nuisance_regressors(motion_params: np.ndarray,
                                 wm_signal: np.ndarray,
                                 csf_signal: np.ndarray,
                                 gs_signal: np.ndarray) -> np.ndarray:
    """
    Construct 24-parameter motion + WM + CSF + GS regressor matrix.
    (Friston 24-parameter model)

    Args:
        motion_params: (T, 6) 6 rigid-body motion parameters
        wm_signal:     (T,)   white matter signal
        csf_signal:    (T,)   CSF signal
        gs_signal:     (T,)   global signal
    Returns:
        regressors: (T, 30) nuisance regressors
    """
    T = motion_params.shape[0]
    # 24 motion parameters: 6 + 6 derivatives + 6 squared + 6 squared derivatives
    mp = motion_params
    mp_deriv = np.vstack([np.zeros((1, 6)), np.diff(mp, axis=0)])
    mp_sq    = mp ** 2
    mp_sq_d  = mp_deriv ** 2
    # Stack all
    regressors = np.hstack([
        mp, mp_deriv, mp_sq, mp_sq_d,              # 24 motion
        wm_signal.reshape(-1, 1),                   # WM
        csf_signal.reshape(-1, 1),                  # CSF
        gs_signal.reshape(-1, 1),                   # Global signal (controversial)
        np.ones((T, 1)),                            # Intercept
        np.arange(T).reshape(-1, 1) / T,           # Linear trend
    ])
    return regressors


def regress_nuisance(roi_ts: np.ndarray,
                     regressors: np.ndarray) -> np.ndarray:
    """
    Remove nuisance regressors from ROI time series using OLS.
    """
    X = regressors
    # OLS: beta = (X^T X)^{-1} X^T Y
    beta = np.linalg.lstsq(X, roi_ts.T, rcond=None)[0]
    residuals = roi_ts.T - X @ beta
    return residuals.T


# ─── EEG / PSG Preprocessing ──────────────────────────────────────────────────

def preprocess_eeg(raw_eeg: np.ndarray,
                   fs: float = 256.0,
                   lo_freq: float = 0.3,
                   hi_freq: float = 50.0,
                   notch_freq: float = 50.0) -> np.ndarray:
    """
    Bandpass filter EEG signal (0.3–50 Hz).

    Args:
        raw_eeg: (N_channels, T) raw EEG
        fs:      sampling frequency
    Returns:
        filtered: (N_channels, T) filtered EEG
    """
    try:
        from scipy.signal import butter, filtfilt, iirnotch
        b_notch, a_notch = iirnotch(notch_freq, Q=30, fs=fs)
        b_bp, a_bp = butter(4, [lo_freq, hi_freq], btype='bandpass', fs=fs)
        filtered = np.zeros_like(raw_eeg)
        for ch in range(raw_eeg.shape[0]):
            sig = filtfilt(b_notch, a_notch, raw_eeg[ch])
            sig = filtfilt(b_bp, a_bp, sig)
            filtered[ch] = sig
        return filtered
    except Exception as e:
        warnings.warn(f"EEG filtering failed: {e}")
        return raw_eeg


def extract_sleep_biomarkers(eeg_filtered: np.ndarray,
                              stage_labels: np.ndarray,
                              fs: float = 256.0) -> Dict[str, float]:
    """
    Extract sleep biomarkers from filtered EEG and staging.

    Returns:
        biomarkers: dict with keys matching SHHS norms
    """
    from scipy import signal as sp_signal

    n_epochs = len(stage_labels)
    epoch_len = int(30 * fs)  # 30-second epochs

    # ── SWA power (0.5–4 Hz) in N3 epochs ───────────────────────────
    n3_mask = (stage_labels == 3)
    swa_powers = []
    for ep_idx in np.where(n3_mask)[0]:
        start = ep_idx * epoch_len
        end   = min(start + epoch_len, eeg_filtered.shape[1])
        if end > start:
            epoch = eeg_filtered[0, start:end]  # F3 channel
            f, psd = sp_signal.welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
            swa_band = psd[(f >= 0.5) & (f <= 4.0)]
            if len(swa_band) > 0:
                swa_powers.append(swa_band.mean())
    swa_power = float(np.mean(swa_powers)) if swa_powers else 0.0

    # ── Sleep spindle density (simplified peak detection) ─────────────
    n2_mask = (stage_labels == 2) | (stage_labels == 3)
    spindle_count = 0
    total_n2_min = n2_mask.sum() * 30 / 60  # in minutes
    for ep_idx in np.where(n2_mask)[0]:
        start = ep_idx * epoch_len
        end   = min(start + epoch_len, eeg_filtered.shape[1])
        if end > start:
            epoch = eeg_filtered[0, start:end]  # C3 channel
            # Sigma envelope
            from scipy.signal import butter, filtfilt, hilbert
            b, a = butter(4, [12, 15], btype='bandpass', fs=fs)
            sigma = filtfilt(b, a, epoch)
            envelope = np.abs(hilbert(sigma))
            threshold = envelope.mean() + 2 * envelope.std()
            # Count spindles (simplified)
            above = (envelope > threshold).astype(int)
            transitions = np.diff(above)
            n_spindles = (transitions == 1).sum()
            spindle_count += n_spindles

    spindle_density = float(spindle_count / (total_n2_min + 1e-8))

    # ── N3 percentage ──────────────────────────────────────────────────
    n3_pct = float(n3_mask.sum() / len(stage_labels) * 100)

    # ── REM fragmentation ──────────────────────────────────────────────
    rem_mask = (stage_labels == 4)
    if rem_mask.sum() > 1:
        transitions = np.diff(rem_mask.astype(int))
        rem_interruptions = (transitions == -1).sum()
        rem_duration_min = rem_mask.sum() * 30 / 60
        rem_frag = float(rem_interruptions / (rem_duration_min + 1e-8))
    else:
        rem_frag = 0.0

    # ── Sleep efficiency ───────────────────────────────────────────────
    sleep_mask = (stage_labels > 0)
    sleep_eff = float(sleep_mask.sum() / len(stage_labels) * 100)

    return {
        'swa_power':       swa_power,
        'spindle_density': spindle_density,
        'n3_pct':          n3_pct,
        'rem_frag':        rem_frag,
        'sleep_eff':       sleep_eff,
        'k_complex_rate':  1.0,    # placeholder — requires dedicated detector
        'sigma_power':     swa_power * 0.3,  # approximate
        'arousal_idx':     float(100 - sleep_eff) / 10,
    }


# ─── DTI Preprocessing ────────────────────────────────────────────────────────

def normalise_sc_matrix(sc_matrix: np.ndarray,
                         method: str = 'log_volume') -> np.ndarray:
    """
    Normalise DTI structural connectivity matrix.

    Args:
        sc_matrix: (N_rois, N_rois) raw streamline count matrix
        method:    'log_volume', 'binary', or 'streamlines'
    Returns:
        sc_norm: (N_rois, N_rois) normalised matrix
    """
    sc = sc_matrix.copy()
    np.fill_diagonal(sc, 0)

    if method == 'log_volume':
        sc_norm = np.log1p(sc)
        # Normalise by node volume (approximate)
        sc_norm = sc_norm / (sc_norm.max() + 1e-8)
    elif method == 'binary':
        sc_norm = (sc > 0).astype(float)
    else:  # raw streamlines
        sc_norm = sc / (sc.max() + 1e-8)

    return sc_norm


def threshold_sc_matrix(sc_matrix: np.ndarray,
                         threshold: float = 0.05) -> np.ndarray:
    """
    Apply proportional threshold to SC matrix (retain top k% connections).
    """
    flat = sc_matrix.flatten()
    cutoff = np.percentile(flat[flat > 0], (1 - threshold) * 100)
    sc_thresholded = sc_matrix.copy()
    sc_thresholded[sc_thresholded < cutoff] = 0
    return sc_thresholded


# ─── Quality Control ──────────────────────────────────────────────────────────

def qc_subject(fmri_motion: np.ndarray,
               eeg_n_hours: float,
               sc_matrix: np.ndarray,
               fd_threshold: float = 0.5,
               min_eeg_hours: float = 6.0,
               min_sc_density: float = 0.1) -> Tuple[bool, str]:
    """
    Quality control check for a single subject.

    Args:
        fmri_motion:   (T,) framewise displacement trace
        eeg_n_hours:   duration of EEG recording in hours
        sc_matrix:     (N_rois, N_rois) structural connectivity matrix
        fd_threshold:  max mean framewise displacement (mm)
        min_eeg_hours: minimum EEG recording duration
        min_sc_density:minimum fraction of non-zero SC entries
    Returns:
        (passed, reason): bool and reason string
    """
    # fMRI motion
    if fmri_motion.mean() > fd_threshold:
        return False, f"High mean FD: {fmri_motion.mean():.3f} > {fd_threshold}"

    # EEG duration
    if eeg_n_hours < min_eeg_hours:
        return False, f"Short EEG: {eeg_n_hours:.1f}h < {min_eeg_hours}h"

    # DTI tractography density
    n_rois = sc_matrix.shape[0]
    density = (sc_matrix > 0).sum() / (n_rois * (n_rois - 1))
    if density < min_sc_density:
        return False, f"Sparse SC: {density:.3f} < {min_sc_density}"

    return True, "QC passed"
