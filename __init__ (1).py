"""
Glymphatic Pathway Graph Transformer (GPG Transformer)
=======================================================
Encodes MRI-derived perivascular space (PVS) features.
Implements SWA-gated dynamic edge weights (Equation 2):
    w_hat_ij(t) = w_ij^struct * sigma(W_g * z_SWA(t) + b_g)

Reference: Equation (2) in PACE-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphSAGE
from torch_geometric.utils import softmax


class SWAGatingModule(nn.Module):
    """
    Computes glymphatic gating weights from SWA power (Equation 2).

    The gate modulates the structural edge weights by a learned
    sigmoidal function of the current slow-wave activity state.
    This implements the empirically established SWS-dependence
    of glymphatic clearance (Iliff et al., 2012; Xie et al., 2013).
    """

    def __init__(self, swa_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(swa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Learnable gating parameters W_g and b_g
        self.W_g = nn.Linear(swa_dim, 1, bias=True)

    def forward(self, w_struct: torch.Tensor,
                z_swa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_struct: (E,) structural edge weights (inverse Euclidean distance)
            z_swa:    (B, swa_dim) slow-wave activity power per subject/epoch
        Returns:
            w_gated: (E,) or (B, E) dynamically gated edge weights
        """
        gate = torch.sigmoid(self.W_g(z_swa))  # (B, 1)
        # Broadcast over edges
        if gate.dim() == 2:
            w_gated = w_struct.unsqueeze(0) * gate  # (B, E)
        else:
            w_gated = w_struct * gate.squeeze()     # (E,)
        return w_gated


class GPGLayer(MessagePassing):
    """
    Single GPG message passing layer with SWA-gated aggregation.
    Uses GraphSAGE-style mean aggregation weighted by glymphatic gates.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.3):
        super().__init__(aggr='add', node_dim=0)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (N, in_dim) PVS node features
            edge_index:  (2, E) edge connectivity
            edge_weight: (E,) or (B, E) SWA-gated weights
        Returns:
            out: (N, out_dim)
        """
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_self(x) + self.lin_neigh(out)
        out = self.norm(self.act(out))
        return self.dropout(out)

    def message(self, x_j: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.unsqueeze(-1) * x_j


class GPGTransformer(nn.Module):
    """
    Full Glymphatic Pathway Graph Transformer.
    Architecture: L=3, H=4, d=64 (from paper Section 3.4)

    Also produces:
      - Glymphatic gating matrix G^glyph (used by CGAT, Equation 6)
      - Latent glymphatic efficiency score z_G (used by Neural-SCM)
    """

    def __init__(self, node_feat_dim: int = 12, hidden_dim: int = 64,
                 n_layers: int = 3, n_heads: int = 4,
                 n_pvs_nodes: int = 24, n_brain_rois: int = 200,
                 dropout: float = 0.3):
        super().__init__()
        self.n_pvs_nodes = n_pvs_nodes
        self.n_brain_rois = n_brain_rois
        self.hidden_dim = hidden_dim

        # SWA gating module
        self.swa_gate = SWAGatingModule(swa_dim=1, hidden_dim=32)

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GPG message passing layers
        self.layers = nn.ModuleList([
            GPGLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Multi-head self-attention for PVS nodes
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Gating matrix projection:
        # Maps PVS embeddings to per-ROI gating weights (N_S x N_C)
        self.gating_proj = nn.Linear(hidden_dim, n_brain_rois)

        # Global pooling for z_G
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                w_struct: torch.Tensor, z_swa: torch.Tensor):
        """
        Args:
            x:           (N_pvs, node_feat_dim) PVS node features
            edge_index:  (2, E) edge connectivity between PVS nodes
            w_struct:    (E,) structural edge weights (inv. Euclidean)
            z_swa:       (B, 1) slow-wave activity per batch subject
        Returns:
            h_G:       (N_pvs, hidden_dim) PVS node embeddings
            G_glyph:   (N_pvs, n_brain_rois) glymphatic gating matrix
            z_G:       (B, hidden_dim) glymphatic efficiency embedding
        """
        # Compute SWA-gated edge weights (Equation 2)
        w_gated = self.swa_gate(w_struct, z_swa)  # (E,) or (B, E)
        if w_gated.dim() == 2:
            # Use mean gate over batch for graph ops
            w_gated_mean = w_gated.mean(0)
        else:
            w_gated_mean = w_gated

        # Input projection
        h = self.input_proj(x)
        h = self.dropout(h)

        # GPG message passing
        for layer in self.layers:
            h = layer(h, edge_index, w_gated_mean)

        # Self-attention over PVS nodes
        h_attn, _ = self.self_attn(
            h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0)
        )
        h = self.norm(h + h_attn.squeeze(0))

        # Glymphatic gating matrix G^glyph ∈ R^(N_pvs x N_rois)
        G_glyph = torch.sigmoid(self.gating_proj(h))  # (N_pvs, N_rois)

        # Global glymphatic embedding z_G
        z_G = self.pool(h.mean(0, keepdim=True))  # (1, hidden_dim)

        return h, G_glyph, z_G
