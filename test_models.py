"""
Cross-Graph Attention Transformer (CGAT)
=========================================
Fuses sleep features and brain connectivity via glymphatic-gated
multi-head cross-attention.

Implements Equations (6) and (7):
    A^cross = softmax( (H^S W_Q)(H^C W_K)^T / sqrt(d_k) ⊙ G^glyph )
    H̃^S    = A^cross · (H^C W_V)

The glymphatic gate G^glyph ∈ R^(N_S × N_C) is produced by the
GPG Transformer and restricts attention to anatomically plausible
sleep-to-brain couplings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlymphaticGatedCrossAttention(nn.Module):
    """
    Multi-head cross-attention with glymphatic gating mask.
    Sleep nodes are queries; brain ROI nodes are keys/values.

    When use_glymphatic_gating=False (ablation), G^glyph ← ones matrix.
    """

    def __init__(self, sleep_dim: int, brain_dim: int,
                 n_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        assert sleep_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = sleep_dim // n_heads
        self.sleep_dim = sleep_dim

        self.W_Q = nn.Linear(sleep_dim, sleep_dim, bias=False)
        self.W_K = nn.Linear(brain_dim, sleep_dim, bias=False)
        self.W_V = nn.Linear(brain_dim, sleep_dim, bias=False)
        self.W_O = nn.Linear(sleep_dim, sleep_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(sleep_dim)

    def forward(self, H_S: torch.Tensor, H_C: torch.Tensor,
                G_glyph: torch.Tensor,
                use_glymphatic_gating: bool = True) -> torch.Tensor:
        """
        Args:
            H_S:     (N_S, sleep_dim) sleep node embeddings
            H_C:     (N_C, brain_dim) brain ROI embeddings
            G_glyph: (N_S, N_C) glymphatic gating matrix from GPG
            use_glymphatic_gating: if False, G_glyph ← ones (ablation)
        Returns:
            H_S_updated: (N_S, sleep_dim) updated sleep embeddings
        """
        N_S = H_S.shape[0]
        N_C = H_C.shape[0]

        # Linear projections
        Q = self.W_Q(H_S).view(N_S, self.n_heads, self.d_k)  # (N_S, H, d_k)
        K = self.W_K(H_C).view(N_C, self.n_heads, self.d_k)  # (N_C, H, d_k)
        V = self.W_V(H_C).view(N_C, self.n_heads, self.d_k)  # (N_C, H, d_k)

        # Attention scores: (N_S, N_C, n_heads)
        Q_ = Q.permute(1, 0, 2)  # (H, N_S, d_k)
        K_ = K.permute(1, 2, 0)  # (H, d_k, N_C)
        scores = torch.bmm(Q_, K_) / math.sqrt(self.d_k)  # (H, N_S, N_C)
        scores = scores.permute(1, 2, 0)  # (N_S, N_C, H)

        # Apply glymphatic gate ⊙ G^glyph  (Equation 6)
        if use_glymphatic_gating:
            gate = G_glyph.unsqueeze(-1)  # (N_S, N_C, 1)
        else:
            # Ablation: no gating — uniform attention across all ROIs
            gate = torch.ones(N_S, N_C, 1, device=H_S.device)
        scores = scores * gate  # (N_S, N_C, H)

        # Softmax over brain ROIs (dim=1)
        attn = F.softmax(scores, dim=1)  # (N_S, N_C, H)
        attn = self.dropout(attn)

        # Weighted sum over brain ROIs  (Equation 7)
        V_ = V.permute(1, 0, 2)                         # (H, N_C, d_k)
        attn_ = attn.permute(2, 0, 1)                   # (H, N_S, N_C)
        H_new = torch.bmm(attn_, V_)                    # (H, N_S, d_k)
        H_new = H_new.permute(1, 0, 2).contiguous()     # (N_S, H, d_k)
        H_new = H_new.view(N_S, self.sleep_dim)         # (N_S, sleep_dim)

        H_new = self.W_O(H_new)

        # Residual + LayerNorm
        H_S_updated = self.norm(H_new + H_S)
        return H_S_updated, attn.mean(-1)  # also return attention map for XAI


class CGATFusion(nn.Module):
    """
    Full Cross-Graph Attention Transformer.
    Fuses H^S (sleep), H^G (glymphatic), H^C (brain FC) via:
      1. Glymphatic-gated cross-attention (sleep queries brain ROIs)
      2. Second self-attention layer with learned fusion token
      3. Output: fused multimodal embedding for downstream tasks
    """

    def __init__(self, sleep_dim: int = 128, brain_dim: int = 256,
                 glyph_dim: int = 64, out_dim: int = 256,
                 n_heads: int = 8, dropout: float = 0.3):
        super().__init__()

        # Project all modalities to common dimension
        self.sleep_proj = nn.Linear(sleep_dim, out_dim)
        self.brain_proj = nn.Linear(brain_dim, out_dim)
        self.glyph_proj = nn.Linear(glyph_dim, out_dim)

        # Cross-attention: sleep queries brain
        self.cross_attn = GlymphaticGatedCrossAttention(
            sleep_dim=out_dim, brain_dim=out_dim,
            n_heads=n_heads, dropout=dropout
        )

        # Fusion token (learnable) — cross-modal bottleneck
        self.fusion_token = nn.Parameter(torch.randn(1, out_dim))
        nn.init.normal_(self.fusion_token, std=0.02)

        # Self-attention over [fusion_token, sleep_nodes, glyph_nodes]
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.out_dim = out_dim

    def forward(self, H_S: torch.Tensor, H_C: torch.Tensor,
                H_G: torch.Tensor, G_glyph: torch.Tensor,
                use_glymphatic_gating: bool = True):
        """
        Args:
            H_S:     (N_S, sleep_dim) sleep node embeddings
            H_C:     (N_C, brain_dim) brain ROI embeddings
            H_G:     (N_pvs, glyph_dim) glymphatic node embeddings
            G_glyph: (N_S, N_C) glymphatic gating matrix
            use_glymphatic_gating: ablation flag
        Returns:
            z_fused:   (out_dim,) global fused representation
            attn_map:  (N_S, N_C) cross-attention map for XAI
        """
        # Project to common dimension
        H_S = self.sleep_proj(H_S)     # (N_S, out_dim)
        H_C = self.brain_proj(H_C)     # (N_C, out_dim)
        H_G = self.glyph_proj(H_G)     # (N_pvs, out_dim)

        # Cross-attention: sleep queries brain (Equations 6-7)
        H_S_updated, attn_map = self.cross_attn(
            H_S, H_C, G_glyph, use_glymphatic_gating
        )

        # Self-attention over [fusion_token, updated_sleep, glyph_nodes]
        tokens = torch.cat([
            self.fusion_token,          # (1, out_dim)
            H_S_updated,               # (N_S, out_dim)
            H_G,                       # (N_pvs, out_dim)
        ], dim=0).unsqueeze(0)         # (1, 1+N_S+N_pvs, out_dim)

        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Fusion token output = global multimodal representation
        z_fused = tokens[0, 0]  # (out_dim,)

        return z_fused, attn_map
