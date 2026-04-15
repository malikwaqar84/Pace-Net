"""
Brain Connectivity Graph Transformer (BCG Transformer)
=======================================================
Encodes stage-stratified dynamic functional connectivity.
Four separate BCGs: G_C^(k) for k in {N1, N2, N3, REM}

Implements Equation (5):
    Attn_ij^BCG = softmax(
        (h_i W_Q)(h_j W_K)^T / sqrt(d_k) * 1[A_ij^SC > 0]
        + lambda * A_ij^FC(k)
    )

Architecture: L=6, H=8, d=256 (paper Section 3.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class BCGAttentionLayer(MessagePassing):
    """
    BCG Transformer attention layer.
    Dual-edge design:
      - Structural edges (DTI tractography): used as hard mask
      - Functional edges (dynamic FC per stage): continuous soft bias
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 8, dropout: float = 0.3,
                 fc_lambda: float = 0.5):
        super().__init__(aggr='add', node_dim=0)
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.out_dim = out_dim
        self.fc_lambda = nn.Parameter(torch.tensor(fc_lambda))

        self.W_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_K = nn.Linear(in_dim, out_dim, bias=False)
        self.W_V = nn.Linear(in_dim, out_dim, bias=False)
        self.W_O = nn.Linear(out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout),
        )

        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                sc_mask: torch.Tensor, fc_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim) ROI node features
            edge_index: (2, E) edges (only anatomically connected ROI pairs)
            sc_mask:    (E,) binary mask — 1 if DTI structural connection exists
            fc_weights: (E,) dynamic FC partial correlation weights for stage k
        Returns:
            out: (N, out_dim)
        """
        Q = self.W_Q(x).view(-1, self.n_heads, self.d_k)
        K = self.W_K(x).view(-1, self.n_heads, self.d_k)
        V = self.W_V(x).view(-1, self.n_heads, self.d_k)

        out = self.propagate(edge_index, Q=Q, K=K, V=V,
                             sc_mask=sc_mask, fc_weights=fc_weights)
        out = out.view(-1, self.out_dim)
        out = self.W_O(out)

        residual = self.residual_proj(x)
        out = self.norm1(out + residual)
        out = self.norm2(out + self.ffn(out))
        return out

    def message(self, Q_i: torch.Tensor, K_j: torch.Tensor,
                V_j: torch.Tensor, sc_mask: torch.Tensor,
                fc_weights: torch.Tensor, index: torch.Tensor,
                ptr: torch.Tensor, size_i: int) -> torch.Tensor:
        scale = self.d_k ** 0.5
        # Base attention score
        attn = (Q_i * K_j).sum(dim=-1) / scale  # (E, n_heads)

        # Apply structural mask (Equation 5: 1[A_ij^SC > 0])
        struct_mask = sc_mask.unsqueeze(-1).float()  # (E, 1)
        attn = attn * struct_mask

        # Add dynamic FC bias: lambda * A_ij^FC(k)
        fc_bias = self.fc_lambda * fc_weights.unsqueeze(-1)  # (E, 1)
        attn = attn + fc_bias

        # Softmax normalisation
        attn = softmax(attn, index, ptr, size_i)
        attn = self.dropout(attn)
        return attn.unsqueeze(-1) * V_j  # (E, n_heads, d_k)


class BCGTransformer(nn.Module):
    """
    Full Brain Connectivity Graph Transformer.
    Processes one sleep stage at a time.
    The four stage outputs are concatenated and then projected.

    Architecture: L=6, H=8, d=256 per stage (paper Section 3.4)
    """

    def __init__(self, node_feat_dim: int = 20, hidden_dim: int = 256,
                 n_layers: int = 6, n_heads: int = 8,
                 dropout: float = 0.3, n_stages: int = 4):
        super().__init__()
        self.n_stages = n_stages
        self.hidden_dim = hidden_dim

        # Input projection (shared across stages)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Transformer layers (shared weights across stages)
        self.layers = nn.ModuleList([
            BCGAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Stage-specific embedding to differentiate N1/N2/N3/REM
        self.stage_embed = nn.Embedding(n_stages, hidden_dim)

        # After concatenating all 4 stage embeddings
        self.fusion_proj = nn.Linear(hidden_dim * n_stages, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_one_stage(self, x: torch.Tensor, edge_index: torch.Tensor,
                          sc_mask: torch.Tensor, fc_weights: torch.Tensor,
                          stage_id: int) -> torch.Tensor:
        """Process a single sleep stage."""
        stage_tensor = torch.tensor(stage_id, device=x.device)
        h = self.input_proj(x) + self.stage_embed(stage_tensor)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, edge_index, sc_mask, fc_weights)
        return h

    def forward(self, x: torch.Tensor,
                edge_index_list: list,
                sc_mask_list: list,
                fc_weights_list: list) -> torch.Tensor:
        """
        Process all four sleep stages and concatenate.

        Args:
            x:               (N_rois, node_feat_dim) ROI node features
            edge_index_list: list of 4 tensors (2, E_k) per stage
            sc_mask_list:    list of 4 tensors (E_k,) structural masks
            fc_weights_list: list of 4 tensors (E_k,) dynamic FC weights
        Returns:
            h_C: (N_rois, hidden_dim) fused BCG embeddings
        """
        stage_outputs = []
        for k in range(self.n_stages):
            h_k = self.forward_one_stage(
                x, edge_index_list[k], sc_mask_list[k],
                fc_weights_list[k], stage_id=k
            )
            stage_outputs.append(h_k)

        # Concatenate along feature dimension: (N, hidden_dim * 4)
        h_concat = torch.cat(stage_outputs, dim=-1)

        # Fuse to (N, hidden_dim)
        h_C = self.norm(self.fusion_proj(h_concat))
        return h_C
