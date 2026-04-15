"""
Sleep Feature Graph Transformer (SFG Transformer)
==================================================
Encodes sleep neurophysiology as a heterogeneous graph.
Node types:
  - EEG channel nodes  (F3, F4, C3, C4, O1, O2, EMG, EOG)
  - Sleep biomarker nodes (SWA power, spindle density, N3%, ...)
Edges: sigma-band spectral coherence (Equation 1 in paper)

Reference: Equation (1) and (4) in PACE-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class BandSpecificAttentionBias(nn.Module):
    """Learnable per-band attention bias b_ij^band (Eq. 4)."""

    def __init__(self, n_bands: int = 5, n_heads: int = 8):
        super().__init__()
        # One bias scalar per (band, head) pair
        self.bias = nn.Parameter(torch.zeros(n_bands, n_heads))
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, band_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_ids: (E,) edge band indices
        Returns:
            bias: (E, n_heads)
        """
        return self.bias[band_ids]


class SFGAttentionLayer(MessagePassing):
    """
    Single SFG Transformer layer with band-specific attention bias.
    Implements Equation (4):
        Attn_ij^SFG = softmax( (h_i W_Q)(h_j W_K)^T / sqrt(d_k) + b_ij^band )
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 8,
                 n_bands: int = 5, dropout: float = 0.3):
        super().__init__(aggr='add', node_dim=0)
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.out_dim = out_dim

        self.W_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_K = nn.Linear(in_dim, out_dim, bias=False)
        self.W_V = nn.Linear(in_dim, out_dim, bias=False)
        self.W_O = nn.Linear(out_dim, out_dim)

        self.band_bias = BandSpecificAttentionBias(n_bands, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, band_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (N, in_dim) node features
            edge_index:  (2, E) edge connectivity
            edge_attr:   (E,) sigma-band coherence weights
            band_ids:    (E,) band index for each edge
        Returns:
            out: (N, out_dim)
        """
        # Multi-head projections
        Q = self.W_Q(x).view(-1, self.n_heads, self.d_k)
        K = self.W_K(x).view(-1, self.n_heads, self.d_k)
        V = self.W_V(x).view(-1, self.n_heads, self.d_k)

        # Band-specific bias per edge
        b_band = self.band_bias(band_ids)  # (E, n_heads)

        out = self.propagate(edge_index, Q=Q, K=K, V=V,
                             edge_attr=edge_attr, b_band=b_band)
        out = out.view(-1, self.out_dim)
        out = self.W_O(out)

        # Residual + LayerNorm
        x_proj = x if x.shape[-1] == self.out_dim else nn.Linear(x.shape[-1], self.out_dim).to(x.device)(x)
        out = self.norm(out + x_proj)
        out = self.norm2(out + self.ffn(out))
        return out

    def message(self, Q_i: torch.Tensor, K_j: torch.Tensor,
                V_j: torch.Tensor, edge_attr: torch.Tensor,
                b_band: torch.Tensor, index: torch.Tensor,
                ptr: torch.Tensor, size_i: int) -> torch.Tensor:
        # (E, n_heads, d_k)
        scale = self.d_k ** 0.5
        attn = (Q_i * K_j).sum(dim=-1) / scale  # (E, n_heads)
        attn = attn + b_band                      # add band bias
        # Scale by edge coherence weight
        attn = attn * edge_attr.unsqueeze(-1)
        attn = softmax(attn, index, ptr, size_i)  # (E, n_heads)
        attn = self.dropout(attn)
        return attn.unsqueeze(-1) * V_j           # (E, n_heads, d_k)


class SFGTransformer(nn.Module):
    """
    Full Sleep Feature Graph Transformer.
    Stacks L SFG attention layers with a learnable stage embedding.

    Architecture: L=4, H=8, d=128 (from paper Section 3.4)
    """

    def __init__(self, node_feat_dim: int = 32, hidden_dim: int = 128,
                 n_layers: int = 4, n_heads: int = 8, n_bands: int = 5,
                 n_stages: int = 5, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers

        # Stage embedding (W, N1, N2, N3, REM) — 5 stages
        self.stage_embedding = nn.Embedding(n_stages, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            SFGAttentionLayer(hidden_dim, hidden_dim, n_heads, n_bands, dropout)
            for _ in range(n_layers)
        ])

        # Output: produces sleep node embeddings h^S
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, band_ids: torch.Tensor,
                stage_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (N, node_feat_dim) node feature matrix
            edge_index:  (2, E) edge index
            edge_attr:   (E,) coherence edge weights (Equation 1)
            band_ids:    (E,) band index per edge
            stage_ids:   (N,) sleep stage ID per node
        Returns:
            h_S: (N, hidden_dim) sleep node embeddings
        """
        # Input projection + stage embedding
        h = self.input_proj(x) + self.stage_embedding(stage_ids)
        h = self.dropout(h)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, band_ids)

        return self.out_proj(h)
