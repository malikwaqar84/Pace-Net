"""
PACE-Net: Full Pathway-Aware Causal Encoder Network
=====================================================
Integrates all components:
  SFGTransformer → GPGTransformer → BCGTransformer →
  HierarchicalDiffPool → CGATFusion → NeuralSCM → outputs

Supports:
  - Full model training
  - Ablation variants (no_sfg, no_glyph_gate, no_causal_loss)
  - XAI hooks (Grad-CAM, attention map extraction)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from models.sfg_transformer import SFGTransformer
from models.gpg_transformer import GPGTransformer
from models.bcg_transformer import BCGTransformer
from models.diffpool import HierarchicalDiffPool
from models.cgat import CGATFusion
from models.neural_scm import NeuralSCM


class PACENet(nn.Module):
    """
    Full PACE-Net model.

    Ablation flags:
        use_sfg:              include Sleep Feature Graph (default True)
        use_gpg:              include Glymphatic Pathway Graph (default True)
        use_glymphatic_gating: use G^glyph in CGAT (default True)
                               False → G^glyph ← ones matrix
        use_neural_scm:       include neural SCM (default True)
        use_cgat:             include CGAT fusion (default True)
    """

    def __init__(self, cfg: dict,
                 use_sfg: bool = True,
                 use_gpg: bool = True,
                 use_glymphatic_gating: bool = True,
                 use_neural_scm: bool = True,
                 use_cgat: bool = True):
        super().__init__()

        self.use_sfg = use_sfg
        self.use_gpg = use_gpg
        self.use_glymphatic_gating = use_glymphatic_gating
        self.use_neural_scm = use_neural_scm
        self.use_cgat = use_cgat

        m = cfg['model']
        d = cfg['data']

        # ── Sleep Feature Graph Transformer ──────────────────────────
        if use_sfg:
            self.sfg_transformer = SFGTransformer(
                node_feat_dim=m['sfg']['node_feat_dim'],
                hidden_dim=m['sfg']['hidden_dim'],
                n_layers=m['sfg']['n_layers'],
                n_heads=m['sfg']['n_heads'],
                n_bands=len(d['eeg_bands']),
                n_stages=d['n_sleep_stages'] + 1,  # +1 for wake
                dropout=m['sfg']['dropout'],
            )
            sfg_out_dim = m['sfg']['hidden_dim']
        else:
            sfg_out_dim = m['sfg']['hidden_dim']  # zero-vector placeholder

        # ── Glymphatic Pathway Graph Transformer ─────────────────────
        if use_gpg:
            self.gpg_transformer = GPGTransformer(
                node_feat_dim=m['gpg']['node_feat_dim'],
                hidden_dim=m['gpg']['hidden_dim'],
                n_layers=m['gpg']['n_layers'],
                n_heads=m['gpg']['n_heads'],
                n_pvs_nodes=d['n_pvs_nodes'],
                n_brain_rois=d['n_rois'],
                dropout=m['gpg']['dropout'],
            )
            gpg_out_dim = m['gpg']['hidden_dim']
        else:
            gpg_out_dim = m['gpg']['hidden_dim']

        # ── Brain Connectivity Graph Transformer ─────────────────────
        self.bcg_transformer = BCGTransformer(
            node_feat_dim=m['bcg']['node_feat_dim'],
            hidden_dim=m['bcg']['hidden_dim'],
            n_layers=m['bcg']['n_layers'],
            n_heads=m['bcg']['n_heads'],
            dropout=m['bcg']['dropout'],
            n_stages=d['n_sleep_stages'],
        )
        bcg_out_dim = m['bcg']['hidden_dim']

        # ── Hierarchical DiffPool ─────────────────────────────────────
        self.diffpool = HierarchicalDiffPool(
            in_dim=bcg_out_dim,
            hidden_dim=m['cgat']['hidden_dim'],
            n_clusters_1=m['diffpool']['n_clusters_1'],
            n_clusters_2=m['diffpool']['n_clusters_2'],
            dropout=m['sfg']['dropout'],
        )
        z_C_raw_dim = m['cgat']['hidden_dim']

        # ── Cross-Graph Attention Transformer ────────────────────────
        if use_cgat:
            self.cgat = CGATFusion(
                sleep_dim=sfg_out_dim,
                brain_dim=bcg_out_dim,
                glyph_dim=gpg_out_dim,
                out_dim=m['cgat']['hidden_dim'],
                n_heads=m['cgat']['n_heads'],
                dropout=m['cgat']['dropout'],
            )

        # ── Neural Structural Causal Model ────────────────────────────
        if use_neural_scm:
            self.neural_scm = NeuralSCM(
                z_S_dim=sfg_out_dim,
                z_G_dim=gpg_out_dim,
                z_C_dim=z_C_raw_dim,
                latent_dim=m['neural_scm']['hidden_dims'][-1],
                n_classes=d['n_classes'],
                dropout=m['sfg']['dropout'],
                t1_feat_dim=64,
                pet_feat_dim=32,
                fmri_feat_dim=z_C_raw_dim,
            )
        else:
            # Simple linear classifier for BCG-only baseline
            self.linear_classifier = nn.Sequential(
                nn.Linear(z_C_raw_dim, 128),
                nn.GELU(),
                nn.Linear(128, d['n_classes'])
            )

        # ── SFG mean pooling (for z_S global vector) ─────────────────
        self.sfg_pool = nn.AdaptiveAvgPool1d(1)

        # ── GPG global pool already in GPGTransformer ─────────────────

        # Zero-embedding fallback for ablation (when SFG/GPG disabled)
        self.register_buffer('zero_sfg', torch.zeros(sfg_out_dim))
        self.register_buffer('zero_gpg', torch.zeros(gpg_out_dim))
        self.register_buffer('ones_ggate',
                             torch.ones(d['n_sleep_channel_nodes']
                                        + d['n_sleep_biomarker_nodes'],
                                        d['n_rois']))

    def forward(self, batch: Dict[str, torch.Tensor],
                return_attention: bool = False,
                s_bar: Optional[torch.Tensor] = None):
        """
        Args:
            batch: dictionary with keys:
              sfg_x, sfg_edge_index, sfg_edge_attr, sfg_band_ids, sfg_stage_ids
              gpg_x, gpg_edge_index, gpg_w_struct, gpg_z_swa
              bcg_x, bcg_edge_index_list, bcg_sc_mask_list, bcg_fc_weights_list
              bcg_adj       — dense adjacency for DiffPool
              x_t1, x_pet, x_fmri  — raw modality projections
            return_attention: whether to return CGAT attention maps
            s_bar: counterfactual normalised sleep (for L_cf)
        Returns:
            logits:     (B, n_classes)
            mmse_pred:  (B, 1) or None
            losses:     dict of component losses
            aux:        dict with z_dict, attention_map, cf_z_C, etc.
        """
        B = batch['bcg_x'].shape[0] if 'bcg_x' in batch else 1

        # ── SFG Transformer ───────────────────────────────────────────
        if self.use_sfg:
            H_S = self.sfg_transformer(
                x=batch['sfg_x'],
                edge_index=batch['sfg_edge_index'],
                edge_attr=batch['sfg_edge_attr'],
                band_ids=batch['sfg_band_ids'],
                stage_ids=batch['sfg_stage_ids'],
            )  # (N_S, sfg_dim)
            # Global sleep embedding (mean pooling)
            z_S_raw = H_S.mean(0, keepdim=True).expand(B, -1)  # (B, sfg_dim)
        else:
            H_S = self.zero_sfg.unsqueeze(0).expand(
                batch['sfg_x'].shape[0], -1)
            z_S_raw = self.zero_sfg.unsqueeze(0).expand(B, -1)

        # ── GPG Transformer ───────────────────────────────────────────
        if self.use_gpg:
            H_G, G_glyph, z_G_raw = self.gpg_transformer(
                x=batch['gpg_x'],
                edge_index=batch['gpg_edge_index'],
                w_struct=batch['gpg_w_struct'],
                z_swa=batch['gpg_z_swa'],
            )
            z_G_raw = z_G_raw.expand(B, -1)
        else:
            N_pvs = batch.get('gpg_x', self.zero_gpg.unsqueeze(0)).shape[0]
            H_G = self.zero_gpg.unsqueeze(0).expand(N_pvs, -1)
            G_glyph = self.ones_ggate
            z_G_raw = self.zero_gpg.unsqueeze(0).expand(B, -1)

        # ── BCG Transformer ───────────────────────────────────────────
        H_C = self.bcg_transformer(
            x=batch['bcg_x'],
            edge_index_list=batch['bcg_edge_index_list'],
            sc_mask_list=batch['bcg_sc_mask_list'],
            fc_weights_list=batch['bcg_fc_weights_list'],
        )  # (N_rois, bcg_dim)

        # ── DiffPool: N_rois → z_C ───────────────────────────────────
        H_C_batch = H_C.unsqueeze(0).expand(B, -1, -1)
        adj_batch = batch['bcg_adj']
        if adj_batch.dim() == 2:
            adj_batch = adj_batch.unsqueeze(0).expand(B, -1, -1)

        z_C_raw, pool_loss = self.diffpool(H_C_batch, adj_batch)
        # z_C_raw: (B, hidden_dim)

        # ── CGAT Fusion ───────────────────────────────────────────────
        attn_map = None
        if self.use_cgat:
            z_fused, attn_map = self.cgat(
                H_S=H_S,
                H_C=H_C,
                H_G=H_G,
                G_glyph=G_glyph,
                use_glymphatic_gating=self.use_glymphatic_gating,
            )
        else:
            z_fused = z_C_raw.mean(0)

        # ── Neural SCM + Classification ───────────────────────────────
        losses = {'pool_loss': pool_loss}

        if self.use_neural_scm:
            logits, mmse_pred, z_dict, cf_z_C, struct_loss = \
                self.neural_scm(
                    z_S_raw=z_S_raw,
                    z_G_raw=z_G_raw,
                    z_C_raw=z_C_raw,
                    x_t1=batch['x_t1'],
                    x_pet=batch['x_pet'],
                    x_fmri=z_C_raw,    # fMRI summary = z_C_raw
                    s_bar=s_bar,
                )
            losses['struct_loss'] = struct_loss
        else:
            # BCG-only baseline
            logits = self.linear_classifier(z_C_raw)
            mmse_pred = None
            z_dict = {'z_C': z_C_raw}
            cf_z_C = None

        aux = {
            'z_dict': z_dict if self.use_neural_scm else None,
            'attention_map': attn_map if return_attention else None,
            'cf_z_C': cf_z_C,
            'H_S': H_S,
            'H_C': H_C,
            'H_G': H_G,
        }

        return logits, mmse_pred, losses, aux
