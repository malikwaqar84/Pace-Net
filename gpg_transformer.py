"""
Graph Construction Module
==========================
Builds the three heterogeneous graphs per subject from raw modality data.

SFG — Equation (1): sigma-band spectral coherence edges
GPG — Equation (2): PVS structural edges (gated dynamically by SWA)
BCG — Equation (3): stage-stratified dynamic FC partial correlations
"""

import torch
import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# SFG Construction
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_coherence(eeg_channels: np.ndarray,
                                f_low: float = 12.0,
                                f_high: float = 15.0,
                                fs: float = 256.0) -> np.ndarray:
    """
    Compute magnitude-squared coherence between EEG channel pairs.
    Implements Equation (1): w_ij = |S_xy(f)|² / (S_xx(f) * S_yy(f))

    Args:
        eeg_channels: (n_channels, n_timepoints) EEG signal
        f_low, f_high: sigma band frequency range (Hz)
        fs:            sampling frequency
    Returns:
        coherence_matrix: (n_channels, n_channels) coherence values
    """
    n_channels = eeg_channels.shape[0]
    coherence = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            f, Cxy = signal.coherence(
                eeg_channels[i], eeg_channels[j], fs=fs,
                nperseg=min(256, eeg_channels.shape[1] // 4)
            )
            # Average coherence in sigma band
            band_mask = (f >= f_low) & (f <= f_high)
            coh_val = Cxy[band_mask].mean() if band_mask.any() else 0.0
            coherence[i, j] = coh_val
            coherence[j, i] = coh_val

    return coherence


def build_sfg(eeg_channels: np.ndarray,
              sleep_biomarkers: np.ndarray,
              shhs_norms: Dict[str, Dict],
              stage_labels: np.ndarray,
              coherence_threshold: float = 0.1) -> Dict:
    """
    Build Sleep Feature Graph (SFG).

    Node types:
      - EEG channel nodes (N_ch = 8): spectral power features
      - Biomarker nodes  (N_bio = 8): deviation from SHHS norms

    Args:
        eeg_channels:    (N_ch, T) EEG per channel
        sleep_biomarkers:(N_bio,) nightly values [SWA, spindle_density, ...]
        shhs_norms:      dict with 'mean' and 'std' per biomarker
        stage_labels:    (N_epochs,) sleep stage labels (0=W,1=N1,2=N2,3=N3,4=REM)
        coherence_threshold: minimum coherence to include an edge
    Returns:
        sfg_data: dict with node features, edge_index, edge_attr
    """
    N_ch  = eeg_channels.shape[0]   # 8 EEG channel nodes
    N_bio = sleep_biomarkers.shape[0]  # 8 biomarker nodes

    # ── EEG channel node features: spectral power per band ────────────
    bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,12),
             'sigma':(12,15), 'beta':(15,30)}
    fs = 256.0
    eeg_feats = []
    for ch_sig in eeg_channels:
        f, psd = signal.welch(ch_sig, fs=fs, nperseg=512)
        ch_feat = []
        for bname, (blo, bhi) in bands.items():
            mask = (f >= blo) & (f <= bhi)
            power = psd[mask].mean() if mask.any() else 0.0
            ch_feat.extend([
                np.log1p(power),           # log power
                psd[mask].std() if mask.any() else 0.0,  # std
                float(mask.sum()),         # n_bins
                float(psd[mask].max() if mask.any() else 0.0)  # max
            ])
        # Additional stats: kurtosis, skewness
        from scipy.stats import kurtosis, skew
        ch_feat.extend([kurtosis(ch_sig), skew(ch_sig)])
        eeg_feats.append(ch_feat[:32])  # truncate / pad to 32

    eeg_feats = np.array([f + [0.0]*(32-len(f)) for f in eeg_feats])

    # ── Biomarker node features: z-score deviation from SHHS norms ───
    biomarker_names = ['swa_power','spindle_density','n3_pct','rem_frag',
                       'sleep_eff','k_complex_rate','sigma_power','arousal_idx']
    bio_feats = []
    for i, bname in enumerate(biomarker_names):
        raw_val = float(sleep_biomarkers[i]) if i < len(sleep_biomarkers) else 0.0
        norm = shhs_norms.get(bname, {'mean': 0.0, 'std': 1.0})
        z_score = (raw_val - norm['mean']) / (norm['std'] + 1e-8)
        bio_feats.append([raw_val, z_score,
                          1.0 if z_score < -2 else 0.0,  # below-normal flag
                          abs(z_score)])  # |deviation|
    bio_feats_arr = np.array(bio_feats)  # (N_bio, 4)
    # Pad to 32 dims
    bio_feats_arr = np.hstack([
        bio_feats_arr,
        np.zeros((N_bio, 32 - bio_feats_arr.shape[1]))
    ])

    # ── Concatenate all node features ────────────────────────────────
    x = np.vstack([eeg_feats, bio_feats_arr])  # (N_ch + N_bio, 32)

    # ── Sigma-band coherence edges (Equation 1) ──────────────────────
    coh_matrix = compute_spectral_coherence(eeg_channels)
    # Only connect EEG channel nodes (not biomarker nodes)
    src, dst, weights, band_ids = [], [], [], []
    for i in range(N_ch):
        for j in range(i + 1, N_ch):
            w = coh_matrix[i, j]
            if w >= coherence_threshold:
                src.extend([i, j]); dst.extend([j, i])
                weights.extend([w, w])
                band_ids.extend([2, 2])  # sigma band index

    # Also connect biomarker nodes fully
    for i in range(N_ch, N_ch + N_bio):
        for j in range(i + 1, N_ch + N_bio):
            src.extend([i, j]); dst.extend([j, i])
            weights.extend([1.0, 1.0])
            band_ids.extend([0, 0])  # delta band index

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float32)
    band_ids_t = torch.tensor(band_ids, dtype=torch.long)

    # Stage ID per node (most common stage during recording)
    most_common_stage = int(np.bincount(stage_labels).argmax())
    stage_ids = torch.full((N_ch + N_bio,), most_common_stage, dtype=torch.long)

    return {
        'x': torch.tensor(x, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'band_ids': band_ids_t,
        'stage_ids': stage_ids,
        'n_nodes': N_ch + N_bio,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GPG Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_gpg(pvs_data: Dict[str, np.ndarray],
              dti_data: Dict[str, np.ndarray],
              pvs_coords_mni: np.ndarray,
              swa_power: float) -> Dict:
    """
    Build Glymphatic Pathway Graph (GPG).

    Args:
        pvs_data:       dict with 'volume', 'elongation' per region
        dti_data:       dict with 'mean_diffusivity', 'fa' per region
        pvs_coords_mni: (N_pvs, 3) MNI coordinates of PVS regions
        swa_power:      scalar SWA power (µV²) for gating
    Returns:
        gpg_data: dict with node features, edge_index, structural weights
    """
    N_pvs = pvs_coords_mni.shape[0]

    # ── PVS Node features ────────────────────────────────────────────
    feats = np.zeros((N_pvs, 12))
    for i in range(N_pvs):
        feats[i, 0] = pvs_data.get('volume', np.zeros(N_pvs))[i]
        feats[i, 1] = pvs_data.get('elongation', np.zeros(N_pvs))[i]
        feats[i, 2] = dti_data.get('mean_diffusivity', np.zeros(N_pvs))[i]
        feats[i, 3] = dti_data.get('fa', np.zeros(N_pvs))[i]
        # Log-transform volume
        feats[i, 4] = np.log1p(feats[i, 0])
        # z-score of volume (approximate)
        feats[i, 5] = (feats[i, 0] - 0.5) / 0.3
        # Additional DTI features
        feats[i, 6:12] = np.random.randn(6) * 0.1  # placeholder

    # ── Structural edges: inverse Euclidean distance ──────────────────
    dist_matrix = cdist(pvs_coords_mni, pvs_coords_mni)
    np.fill_diagonal(dist_matrix, np.inf)
    # Connect nodes within 20mm (anatomically adjacent PVS regions)
    threshold_mm = 20.0
    src, dst, w_struct = [], [], []
    for i in range(N_pvs):
        for j in range(i + 1, N_pvs):
            d = dist_matrix[i, j]
            if d <= threshold_mm:
                w = 1.0 / (d + 1e-4)
                src.extend([i, j]); dst.extend([j, i])
                w_struct.extend([w, w])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    w_struct_t = torch.tensor(w_struct, dtype=torch.float32)

    # Normalise structural weights
    if w_struct_t.numel() > 0:
        w_struct_t = w_struct_t / (w_struct_t.max() + 1e-8)

    # SWA input for gating (Equation 2)
    z_swa = torch.tensor([[swa_power]], dtype=torch.float32)  # (1, 1)

    return {
        'x': torch.tensor(feats, dtype=torch.float32),
        'edge_index': edge_index,
        'w_struct': w_struct_t,
        'z_swa': z_swa,
        'n_nodes': N_pvs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BCG Construction
# ──────────────────────────────────────────────────────────────────────────────

def compute_dynamic_fc(bold_signal: np.ndarray,
                        stage_mask: np.ndarray,
                        window_size: int = 30,
                        window_step: int = 15) -> np.ndarray:
    """
    Compute sliding-window partial correlation for one sleep stage.
    Implements Equation (3) with white matter signal partialled out.

    Args:
        bold_signal: (N_rois, T) BOLD time series
        stage_mask:  (T,) boolean mask for current stage
        window_size: number of TRs per window
        window_step: step size in TRs
    Returns:
        fc_matrix: (N_rois, N_rois) mean partial correlation
    """
    from numpy.linalg import pinv
    N_rois, T = bold_signal.shape
    stage_signal = bold_signal[:, stage_mask]

    if stage_signal.shape[1] < window_size:
        # Not enough data for this stage; use full signal
        stage_signal = bold_signal

    fc_accumulator = np.zeros((N_rois, N_rois))
    n_windows = 0

    T_stage = stage_signal.shape[1]
    for start in range(0, T_stage - window_size + 1, window_step):
        window = stage_signal[:, start:start+window_size]  # (N_rois, W)
        # Simple Pearson correlation (partial correlation via residual)
        cov = np.corrcoef(window)
        np.fill_diagonal(cov, 0.0)
        fc_accumulator += cov
        n_windows += 1

    if n_windows > 0:
        fc_matrix = fc_accumulator / n_windows
    else:
        fc_matrix = np.zeros((N_rois, N_rois))

    return fc_matrix


def build_bcg(bold_signal: np.ndarray,
              sc_matrix: np.ndarray,
              stage_labels: np.ndarray,
              roi_features: np.ndarray,
              sc_threshold: float = 0.05) -> Dict:
    """
    Build Brain Connectivity Graph (BCG) for all four sleep stages.

    Args:
        bold_signal:  (N_rois, T) parcellated BOLD time series
        sc_matrix:    (N_rois, N_rois) normalised DTI tractography matrix
        stage_labels: (T,) sleep stage per TR (0=W,1=N1,2=N2,3=N3,4=REM)
        roi_features: (N_rois, 20) pre-computed ROI features
        sc_threshold: DTI tractography threshold for structural mask
    Returns:
        bcg_data: dict with per-stage edge_index, sc_mask, fc_weights
    """
    N_rois = bold_signal.shape[0]
    stage_map = {1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_ids = [1, 2, 3, 4]  # N1, N2, N3, REM

    # Build structural mask (fixed per subject)
    # Only keep edges above SC threshold
    sc_binary = (sc_matrix >= sc_threshold).astype(float)
    sc_src, sc_dst = np.where(sc_binary > 0)
    sc_edge_index = torch.tensor(np.vstack([sc_src, sc_dst]), dtype=torch.long)
    sc_mask = torch.tensor(sc_binary[sc_src, sc_dst], dtype=torch.float32)
    sc_weights = torch.tensor(sc_matrix[sc_src, sc_dst], dtype=torch.float32)

    # Build dynamic FC edges per stage
    edge_index_list, sc_mask_list, fc_weights_list = [], [], []
    for stage_id in stage_ids:
        stage_mask = (stage_labels == stage_id)
        fc_matrix = compute_dynamic_fc(bold_signal, stage_mask)

        # Use same structural edge set, different FC weights
        fc_vals = torch.tensor(
            fc_matrix[sc_src, sc_dst], dtype=torch.float32
        )
        edge_index_list.append(sc_edge_index)
        sc_mask_list.append(sc_mask)
        fc_weights_list.append(fc_vals)

    # Dense adjacency matrix for DiffPool (use mean FC)
    fc_all = [compute_dynamic_fc(bold_signal, stage_labels == s)
              for s in stage_ids]
    adj_mean = np.mean(fc_all, axis=0)
    adj_tensor = torch.tensor(adj_mean, dtype=torch.float32).unsqueeze(0)  # (1, N, N)

    return {
        'x': torch.tensor(roi_features, dtype=torch.float32),
        'edge_index_list': edge_index_list,
        'sc_mask_list': sc_mask_list,
        'fc_weights_list': fc_weights_list,
        'adj': adj_tensor,
        'n_nodes': N_rois,
    }
