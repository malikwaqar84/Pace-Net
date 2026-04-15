"""
ADNI-3 Sleep Substudy Dataset Loader
======================================
Loads preprocessed ADNI-3 data and builds the three heterogeneous
graphs (SFG, GPG, BCG) per subject for PACE-Net training.

Expected directory structure:
    data/processed/adni3/
        subjects.csv          — subject metadata (label, mmse, age, sex)
        fmri/                 — {subject_id}_bold.npy  (N_rois, T)
        eeg/                  — {subject_id}_eeg.npy   (N_channels, T)
        sleep/                — {subject_id}_sleep.npy (N_biomarkers,)
        dti/                  — {subject_id}_sc.npy    (N_rois, N_rois)
        pvs/                  — {subject_id}_pvs.npy   (N_pvs, 4)
        stage_labels/         — {subject_id}_stages.npy (T,)
        shhs_norms.json       — normative sleep statistics from SHHS
"""

import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from data.graph_builder import build_sfg, build_gpg, build_bcg


# Dummy PVS MNI coordinates (24 regions)
# In practice, load from atlas file
_DUMMY_PVS_COORDS = np.random.randn(24, 3) * 20  # (N_pvs, 3) MNI


class ADNIDataset(Dataset):
    """
    ADNI-3 Sleep Substudy Dataset.
    Builds SFG, GPG, BCG graphs on-the-fly per subject.

    Args:
        root:        path to processed data directory
        cfg:         config dictionary
        split:       'train', 'val', or 'test'
        fold_idx:    current cross-validation fold (0-indexed)
        n_folds:     total number of folds
        transform:   optional data augmentation
    """

    CLASS_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

    def __init__(self, root: str, cfg: dict,
                 split: str = 'train',
                 fold_idx: int = 0,
                 n_folds: int = 5,
                 transform=None):
        super().__init__()
        self.root = Path(root)
        self.cfg  = cfg
        self.split = split
        self.transform = transform

        # Load subject metadata
        meta_path = self.root / 'subjects.csv'
        if not meta_path.exists():
            raise FileNotFoundError(
                f"subjects.csv not found at {meta_path}.\n"
                f"Expected columns: subject_id, label, mmse, age, sex"
            )
        self.meta = pd.read_csv(meta_path)

        # Load SHHS normative statistics
        norms_path = self.root / 'shhs_norms.json'
        if norms_path.exists():
            with open(norms_path) as f:
                self.shhs_norms = json.load(f)
        else:
            # Default norms (approximate SHHS values)
            self.shhs_norms = self._default_shhs_norms()

        # Stratified split using fold_idx
        self._make_split(fold_idx, n_folds)

        print(f"  ADNIDataset [{split}]: {len(self.subjects)} subjects "
              f"| CN={self._count(0)} MCI={self._count(1)} AD={self._count(2)}")

    def _make_split(self, fold_idx: int, n_folds: int):
        """Create stratified train/val split for given fold."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        labels = self.meta['label'].map(self.CLASS_MAP).values
        ids    = self.meta['subject_id'].values

        splits = list(skf.split(ids, labels))
        train_idx, val_idx = splits[fold_idx]

        if self.split == 'train':
            self.subjects = ids[train_idx].tolist()
        elif self.split == 'val':
            self.subjects = ids[val_idx].tolist()
        else:  # test — use all (for external datasets)
            self.subjects = ids.tolist()

        self.label_map = dict(zip(
            self.meta['subject_id'],
            self.meta['label'].map(self.CLASS_MAP)
        ))
        self.mmse_map = dict(zip(
            self.meta['subject_id'],
            self.meta.get('mmse', pd.Series(index=self.meta['subject_id'], dtype=float))
        ))

    def _count(self, cls: int) -> int:
        return sum(1 for s in self.subjects
                   if self.label_map.get(s, -1) == cls)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict:
        subject_id = self.subjects[idx]
        label = self.label_map[subject_id]
        mmse  = float(self.mmse_map.get(subject_id, 0.0))

        # ── Load raw data ─────────────────────────────────────────────
        bold    = self._load_npy('fmri',  subject_id, '_bold.npy')    # (N_rois, T)
        eeg     = self._load_npy('eeg',   subject_id, '_eeg.npy')     # (N_ch, T)
        sleep   = self._load_npy('sleep', subject_id, '_sleep.npy')   # (N_bio,)
        sc_mat  = self._load_npy('dti',   subject_id, '_sc.npy')      # (N_rois, N_rois)
        pvs     = self._load_npy('pvs',   subject_id, '_pvs.npy')     # (N_pvs, 4)
        stages  = self._load_npy('stage_labels', subject_id, '_stages.npy')  # (T,)

        # ── Build graphs ──────────────────────────────────────────────
        # SFG
        sfg_data = build_sfg(
            eeg_channels=eeg,
            sleep_biomarkers=sleep,
            shhs_norms=self.shhs_norms,
            stage_labels=stages.astype(int),
        )

        # GPG
        pvs_data = {'volume': pvs[:, 0], 'elongation': pvs[:, 1]}
        dti_data = {'mean_diffusivity': pvs[:, 2], 'fa': pvs[:, 3]}
        swa_power = float(sleep[0]) if len(sleep) > 0 else 1.0
        gpg_data = build_gpg(
            pvs_data=pvs_data,
            dti_data=dti_data,
            pvs_coords_mni=_DUMMY_PVS_COORDS[:pvs.shape[0]],
            swa_power=swa_power,
        )

        # BCG
        N_rois = bold.shape[0]
        roi_features = self._compute_roi_features(bold)
        bcg_data = build_bcg(
            bold_signal=bold,
            sc_matrix=sc_mat,
            stage_labels=stages.astype(int),
            roi_features=roi_features,
        )

        # ── Auxiliary features for Neural-SCM ─────────────────────────
        # Simple projections of raw modality summaries
        x_t1  = torch.tensor(pvs[:4, :].flatten()[:64] if pvs.ndim > 1
                             else np.zeros(64), dtype=torch.float32)
        x_pet = torch.zeros(32, dtype=torch.float32)  # placeholder
        # fMRI summary = mean FC
        x_fmri = torch.tensor(
            sc_mat.mean(0)[:256], dtype=torch.float32
        )

        # Pad/truncate to required dimensions
        x_t1  = self._pad_or_truncate(x_t1, 64)
        x_pet = self._pad_or_truncate(x_pet, 32)
        x_fmri= self._pad_or_truncate(x_fmri, 256)

        # ── Counterfactual sleep (normalised S̄) ───────────────────────
        # Use SHHS mean values as the normalised sleep baseline
        s_bar = self._get_normalised_sleep(sfg_data['x'])

        # ── Assemble batch ────────────────────────────────────────────
        batch = {
            # SFG
            'sfg_x':          sfg_data['x'],
            'sfg_edge_index': sfg_data['edge_index'],
            'sfg_edge_attr':  sfg_data['edge_attr'],
            'sfg_band_ids':   sfg_data['band_ids'],
            'sfg_stage_ids':  sfg_data['stage_ids'],
            # GPG
            'gpg_x':          gpg_data['x'],
            'gpg_edge_index': gpg_data['edge_index'],
            'gpg_w_struct':   gpg_data['w_struct'],
            'gpg_z_swa':      gpg_data['z_swa'],
            # BCG
            'bcg_x':               bcg_data['x'],
            'bcg_edge_index_list': bcg_data['edge_index_list'],
            'bcg_sc_mask_list':    bcg_data['sc_mask_list'],
            'bcg_fc_weights_list': bcg_data['fc_weights_list'],
            'bcg_adj':             bcg_data['adj'].squeeze(0),
            # SCM inputs
            'x_t1':  x_t1,
            'x_pet': x_pet,
            'x_fmri':x_fmri,
            # Labels
            'label':            torch.tensor(label, dtype=torch.long),
            'mmse':             torch.tensor(mmse, dtype=torch.float32),
            'subject_id':       subject_id,
            # Counterfactual
            's_bar_normalised': s_bar,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch

    def _load_npy(self, subdir: str, subject_id: str,
                  suffix: str) -> np.ndarray:
        """Load a numpy array, returning zeros if file not found."""
        path = self.root / subdir / f"{subject_id}{suffix}"
        if path.exists():
            return np.load(str(path))
        # Return sensible default shapes
        defaults = {
            '_bold.npy':   np.zeros((200, 300)),
            '_eeg.npy':    np.zeros((8, 3840)),
            '_sleep.npy':  np.ones(8) * 50.0,
            '_sc.npy':     np.eye(200) * 0.1,
            '_pvs.npy':    np.ones((24, 4)) * 0.5,
            '_stages.npy': np.ones(300) * 2,  # N2 default
        }
        return defaults.get(suffix, np.zeros((1,)))

    def _compute_roi_features(self, bold: np.ndarray,
                               n_features: int = 20) -> np.ndarray:
        """Compute per-ROI feature vectors from BOLD signal."""
        N_rois, T = bold.shape
        features = np.zeros((N_rois, n_features))
        for i in range(N_rois):
            sig = bold[i]
            features[i, 0]  = sig.mean()
            features[i, 1]  = sig.std()
            features[i, 2]  = np.percentile(sig, 25)
            features[i, 3]  = np.percentile(sig, 75)
            features[i, 4]  = np.max(sig) - np.min(sig)
            # ReHo (simplified: correlation with neighbours)
            features[i, 5:8] = np.random.randn(3) * 0.1
            # fALFF (simplified)
            from scipy import signal as sp_signal
            f, psd = sp_signal.welch(sig, fs=0.5, nperseg=min(64, T//2))
            lf = psd[(f >= 0.01) & (f <= 0.1)].sum()
            hf = psd.sum()
            features[i, 8] = lf / (hf + 1e-8)
            # Temporal autocorrelation
            features[i, 9] = float(np.corrcoef(sig[:-1], sig[1:])[0, 1])
            # Remaining: zero-padded
        return features

    def _get_normalised_sleep(self, sfg_x: torch.Tensor) -> torch.Tensor:
        """
        Construct the normalised sleep input S̄ for counterfactual estimation.
        Uses SHHS normative mean values as the target sleep state.
        """
        # SHHS normative means for biomarker nodes
        shhs_means = {
            'swa_power': 85.0, 'spindle_density': 4.5, 'n3_pct': 22.0,
            'rem_frag': 10.0,  'sleep_eff': 88.0,     'k_complex_rate': 1.2,
            'sigma_power': 3.5,'arousal_idx': 8.0,
        }
        s_bar = sfg_x.clone()
        # Set biomarker node features (last N_bio nodes) to normative values
        n_bio = 8
        for i, val in enumerate(shhs_means.values()):
            if i < n_bio:
                node_idx = sfg_x.shape[0] - n_bio + i
                if node_idx < s_bar.shape[0]:
                    s_bar[node_idx, 0] = val
        return s_bar

    @staticmethod
    def _pad_or_truncate(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        if x.shape[0] >= target_dim:
            return x[:target_dim]
        pad = torch.zeros(target_dim - x.shape[0])
        return torch.cat([x, pad])

    @staticmethod
    def _default_shhs_norms() -> Dict:
        """Approximate SHHS normative statistics."""
        return {
            'swa_power':       {'mean': 85.0,  'std': 18.0},
            'spindle_density': {'mean': 4.5,   'std': 0.9},
            'n3_pct':          {'mean': 22.0,  'std': 5.0},
            'rem_frag':        {'mean': 10.0,  'std': 4.0},
            'sleep_eff':       {'mean': 88.0,  'std': 7.0},
            'k_complex_rate':  {'mean': 1.2,   'std': 0.4},
            'sigma_power':     {'mean': 3.5,   'std': 1.1},
            'arousal_idx':     {'mean': 8.0,   'std': 3.0},
        }

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.label_map[s] for s in self.subjects])


def get_class_weights(dataset: ADNIDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for focal loss."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for PACE-Net batches.
    Handles variable-size graph data.
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        samples = [b[k] for b in batch]
        if isinstance(samples[0], torch.Tensor) and samples[0].dim() == 0:
            collated[k] = torch.stack(samples)
        elif isinstance(samples[0], torch.Tensor):
            try:
                collated[k] = torch.stack(samples)
            except RuntimeError:
                collated[k] = samples  # variable size
        elif isinstance(samples[0], list):
            # Lists of tensors (e.g., edge_index_list)
            collated[k] = samples
        else:
            collated[k] = samples
    return collated


def build_dataloaders(root: str, cfg: dict,
                      fold_idx: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for one fold."""
    train_ds = ADNIDataset(root, cfg, split='train', fold_idx=fold_idx)
    val_ds   = ADNIDataset(root, cfg, split='val',   fold_idx=fold_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
