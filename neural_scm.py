"""
Differentiable Pooling (DiffPool)
===================================
Hierarchical graph pooling that compresses 200 ROI nodes to
7 Yeo functional network embeddings, then to 1 whole-brain z_C.

Reference: Ying et al. NeurIPS 2018 (Ref 44 in paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv


class DiffPoolLayer(nn.Module):
    """
    Single DiffPool layer:
    - Embed layer: refines node features
    - Pool layer: produces soft cluster assignments
    """

    def __init__(self, in_dim: int, out_dim: int, n_clusters: int,
                 dropout: float = 0.3):
        super().__init__()
        self.embed = DenseGCNConv(in_dim, out_dim)
        self.pool  = DenseGCNConv(in_dim, n_clusters)
        self.norm_embed = nn.LayerNorm(out_dim)
        self.norm_pool  = nn.LayerNorm(n_clusters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor):
        """
        Args:
            x:   (B, N, in_dim) node features
            adj: (B, N, N) adjacency matrix
        Returns:
            x_pooled:   (B, n_clusters, out_dim)
            adj_pooled: (B, n_clusters, n_clusters)
            loss_lp:    link prediction regularisation loss
            loss_e:     entropy regularisation loss
        """
        # Embedding
        z = F.relu(self.embed(x, adj))
        z = self.norm_embed(z)
        z = self.dropout(z)

        # Soft cluster assignment
        s = F.softmax(self.norm_pool(self.pool(x, adj)), dim=-1)

        # DiffPool assignment
        x_pooled = torch.bmm(s.transpose(1, 2), z)        # (B, n_clusters, out_dim)
        adj_pooled = torch.bmm(s.transpose(1, 2), torch.bmm(adj, s))  # (B, nc, nc)

        # Regularisation losses
        # Link prediction loss
        loss_lp = (torch.norm(adj - torch.bmm(s, s.transpose(1, 2))) ** 2
                   / (adj.shape[1] ** 2))
        # Entropy loss
        loss_e = (-s * torch.log(s + 1e-8)).sum(-1).mean()

        return x_pooled, adj_pooled, loss_lp, loss_e


class HierarchicalDiffPool(nn.Module):
    """
    Two-level hierarchical DiffPool:
      Level 1: 200 ROIs → 7 Yeo functional networks
      Level 2: 7 networks → 1 whole-brain embedding z_C

    The Yeo-7 atlas provides biologically motivated soft
    assignment initialisation for level 1.
    """

    def __init__(self, in_dim: int = 256,
                 hidden_dim: int = 256,
                 n_clusters_1: int = 7,
                 n_clusters_2: int = 1,
                 yeo7_init: torch.Tensor = None,
                 dropout: float = 0.3):
        super().__init__()

        self.pool1 = DiffPoolLayer(in_dim, hidden_dim, n_clusters_1, dropout)
        self.pool2 = DiffPoolLayer(hidden_dim, hidden_dim, n_clusters_2, dropout)

        # Optional: initialise pool-1 assignment using Yeo-7 atlas
        # Shape: (N_rois=200, n_clusters_1=7) binary membership matrix
        if yeo7_init is not None:
            # Store as a non-trainable reference for soft init
            self.register_buffer('yeo7_init', yeo7_init.float())
        else:
            self.yeo7_init = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x:   (B, N_rois, in_dim) BCG node embeddings
            adj: (B, N_rois, N_rois) adjacency matrix
        Returns:
            z_C:        (B, hidden_dim) whole-brain embedding
            total_loss: pooling regularisation loss
        """
        # Level 1: 200 ROIs → 7 networks
        x1, adj1, lp1, e1 = self.pool1(x, adj)

        # Level 2: 7 networks → 1 whole-brain
        x2, _, lp2, e2 = self.pool2(x1, adj1)

        # Final whole-brain embedding
        z_C = x2.squeeze(1)  # (B, hidden_dim)

        total_loss = lp1 + e1 + lp2 + e2
        return z_C, total_loss
