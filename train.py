"""
Neural Structural Causal Model (Neural-SCM)
============================================
Biologically-grounded latent variable model.
Enforces the ordering: z_S → z_G → z_A → z_C → Y

Structural Equations (8)-(10):
    z_G = f_G(z_S, x_T1;  θ_G) + ε_G
    z_A = f_A(z_G, x_PET; θ_A) + ε_A
    z_C = f_C(z_A, x_fMRI;θ_C) + ε_C

Losses (11)-(13):
    L_struct = Σ ||z_k - f_k(z_pa(k), x_k)||²
    L_cf     = ||z_C^do(S=S̄) - z̄_C^CN||²

IMPORTANT: This is a structured DISCRIMINATIVE model.
It does NOT prove causation. The ordering reflects
biological hypothesis, not controlled experimental evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StructuralEquation(nn.Module):
    """
    A single structural equation: z_k = f_k(z_pa(k), x_k) + ε_k
    Implemented as a 2-layer MLP.
    """

    def __init__(self, parent_dim: int, obs_dim: int,
                 out_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        in_dim = parent_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        # Noise scale parameter
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, z_parent: torch.Tensor,
                x_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_parent: (B, parent_dim) parent latent variable
            x_obs:    (B, obs_dim)   observed features for this stage
        Returns:
            z_k:   (B, out_dim) output latent variable
            noise: (B, out_dim) noise term ε_k (reparameterised)
        """
        z_in = torch.cat([z_parent, x_obs], dim=-1)
        z_k_mean = self.net(z_in)
        # Reparameterisation: z_k = mean + σ * ε
        sigma = torch.exp(self.log_sigma).clamp(min=1e-4, max=1.0)
        noise = torch.randn_like(z_k_mean) * sigma
        z_k = z_k_mean + noise
        return z_k, noise


class NeuralSCM(nn.Module):
    """
    Full biologically-grounded Neural Structural Causal Model.

    Latent variables:
      z_S — sleep disruption (from SFG Transformer)
      z_G — glymphatic efficiency (from GPG Transformer)
      z_A — amyloid burden proxy (from structural equation)
      z_C — FC disruption (from BCG + DiffPool)

    Outputs:
      - Classification logits (CN/MCI/AD)
      - MMSE regression (auxiliary)
      - Counterfactual z_C^do(S=S̄)
    """

    def __init__(self, z_S_dim: int = 128, z_G_dim: int = 64,
                 z_C_dim: int = 256, latent_dim: int = 128,
                 n_classes: int = 3, dropout: float = 0.3,
                 # Observed feature dims (projections from raw modalities)
                 t1_feat_dim: int = 64,
                 pet_feat_dim: int = 32,
                 fmri_feat_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Project raw modality features to structural equation inputs
        self.t1_proj   = nn.Linear(t1_feat_dim, latent_dim)
        self.pet_proj  = nn.Linear(pet_feat_dim, latent_dim)
        self.fmri_proj = nn.Linear(fmri_feat_dim, latent_dim)

        # Project z_S (from SFG) to latent space
        self.z_S_proj = nn.Linear(z_S_dim, latent_dim)

        # Project z_G (from GPG) to latent space
        self.z_G_proj = nn.Linear(z_G_dim, latent_dim)

        # Project z_C (from BCG+DiffPool) to latent space
        self.z_C_proj = nn.Linear(z_C_dim, latent_dim)

        # Structural equation: z_G = f_G(z_S, x_T1) — Equation (8)
        self.eq_G = StructuralEquation(
            parent_dim=latent_dim, obs_dim=latent_dim,
            out_dim=latent_dim, hidden_dim=latent_dim * 2, dropout=dropout
        )

        # Structural equation: z_A = f_A(z_G, x_PET) — Equation (9)
        self.eq_A = StructuralEquation(
            parent_dim=latent_dim, obs_dim=latent_dim,
            out_dim=latent_dim, hidden_dim=latent_dim * 2, dropout=dropout
        )

        # Structural equation: z_C = f_C(z_A, x_fMRI) — Equation (10)
        self.eq_C = StructuralEquation(
            parent_dim=latent_dim, obs_dim=latent_dim,
            out_dim=latent_dim, hidden_dim=latent_dim * 2, dropout=dropout
        )

        # Final classifier on [z_S, z_G, z_A, z_C, z_fused]
        combined_dim = latent_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # Auxiliary MMSE regression
        self.mmse_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # CN population mean z_C (updated during training for L_cf)
        self.register_buffer('cn_z_C_mean',
                             torch.zeros(latent_dim))

        # Counterfactual: project z_C under do(S=S̄)
        self.cf_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_S_raw: torch.Tensor,
                z_G_raw: torch.Tensor,
                z_C_raw: torch.Tensor,
                x_t1: torch.Tensor,
                x_pet: torch.Tensor,
                x_fmri: torch.Tensor,
                s_bar: Optional[torch.Tensor] = None):
        """
        Args:
            z_S_raw:  (B, z_S_dim) sleep disruption embedding (from SFG)
            z_G_raw:  (B, z_G_dim) glymphatic efficiency (from GPG)
            z_C_raw:  (B, z_C_dim) FC disruption (from BCG+DiffPool)
            x_t1:     (B, t1_feat_dim) T1-MRI features
            x_pet:    (B, pet_feat_dim) PET proxy features
            x_fmri:   (B, fmri_feat_dim) fMRI summary features
            s_bar:    (B, z_S_dim) counterfactual sleep (normalised S̄)
                      if None, counterfactual is not computed
        Returns:
            logits:       (B, n_classes)
            mmse_pred:    (B, 1)
            z_dict:       dict of all latent variables
            cf_z_C:       (B, latent_dim) counterfactual z_C (or None)
            struct_loss:  scalar structural consistency loss L_struct
        """
        # Project to latent space
        z_S = self.z_S_proj(z_S_raw)      # (B, D)
        z_G_obs = self.z_G_proj(z_G_raw)  # (B, D) - from GPG
        z_C_obs = self.z_C_proj(z_C_raw)  # (B, D) - from BCG+DiffPool

        x_t1_  = self.t1_proj(x_t1)
        x_pet_ = self.pet_proj(x_pet)
        x_fmri_= self.fmri_proj(x_fmri)

        # ── Structural equations ──────────────────────────────────────
        # Eq. 8: z_G_pred = f_G(z_S, x_T1)
        z_G_pred, _ = self.eq_G(z_S, x_t1_)

        # Eq. 9: z_A = f_A(z_G, x_PET)
        z_A, _ = self.eq_A(z_G_pred, x_pet_)

        # Eq. 10: z_C_pred = f_C(z_A, x_fMRI)
        z_C_pred, _ = self.eq_C(z_A, x_fmri_)

        # ── Structural consistency loss L_struct (Eq. 11) ────────────
        loss_G = F.mse_loss(z_G_pred, z_G_obs.detach())
        loss_C = F.mse_loss(z_C_pred, z_C_obs.detach())
        struct_loss = loss_G + loss_C

        # Use observed z_C for classification (more informative)
        z_C = z_C_obs

        # ── Classification and regression ────────────────────────────
        z_all = torch.cat([z_S, z_G_pred, z_A, z_C], dim=-1)
        z_all = self.dropout(z_all)
        logits    = self.classifier(z_all)
        mmse_pred = self.mmse_head(z_all)

        # ── Counterfactual FC estimation (L_cf, Eq. 13) ──────────────
        cf_z_C = None
        if s_bar is not None:
            z_S_cf = self.z_S_proj(s_bar)
            z_G_cf, _ = self.eq_G(z_S_cf, x_t1_)
            z_A_cf, _ = self.eq_A(z_G_cf, x_pet_)
            cf_z_C, _ = self.eq_C(z_A_cf, x_fmri_)
            cf_z_C = self.cf_proj(cf_z_C)

        z_dict = {
            'z_S': z_S, 'z_G': z_G_pred,
            'z_A': z_A, 'z_C': z_C,
            'z_G_obs': z_G_obs, 'z_C_obs': z_C_obs
        }
        return logits, mmse_pred, z_dict, cf_z_C, struct_loss

    @torch.no_grad()
    def update_cn_mean(self, z_C_cn: torch.Tensor, momentum: float = 0.99):
        """Update running mean of CN population z_C (for L_cf)."""
        self.cn_z_C_mean = (momentum * self.cn_z_C_mean
                            + (1 - momentum) * z_C_cn.mean(0).detach())
