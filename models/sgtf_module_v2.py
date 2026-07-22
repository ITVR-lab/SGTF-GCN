"""
SGTFModuleV2 — decoupled version.

Unlike SGTFModule (v1), this module does NOT fuse A_gap and A_ljp into a single
adjacency. It returns them separately so each stream can be processed independently
in the tri-stream GCN unit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sgtf_module import GAPModule, LJPCache


class SGTFModuleV2(nn.Module):
    """Semantics-Guided Topology module v2 (decoupled streams).

    Returns A_gap (N, V, V) and A_ljp (N, V, V) separately.
    No weighted fusion is performed here; each stream is handled
    independently in unit_ctrgcn_teacher_tri.

    Args:
        clip_dim     : CLIP embedding dimension (512 for ViT-B/32).
        feat_channels: C of input skeleton features at this block.
        num_classes  : K — number of action classes.
        num_joints   : V — number of skeleton joints.
        d_h          : attention projection dim for GAPModule.
        mlp_hidden   : hidden dim in GAPModule MLP.
    """

    def __init__(self,
                 clip_dim: int,
                 feat_channels: int,
                 num_classes: int,
                 num_joints: int,
                 d_h: int = 64,
                 mlp_hidden: int = 256):
        super().__init__()
        self.gap = GAPModule(clip_dim, feat_channels, d_h, mlp_hidden)
        self.ljp_cache = LJPCache(num_classes, num_joints)

    def forward(self,
                x: torch.Tensor,
                t_sem: torch.Tensor,
                labels: torch.Tensor):
        """
        Args:
            x:      (N, C, T, V)
            t_sem:  (N, clip_dim) CLIP embeddings for this batch
            labels: (N,) ground-truth class indices (long)
        Returns:
            A_gap: (N, V, V)
            A_ljp: (N, V, V)
        """
        A_gap = torch.nan_to_num(
            self.gap(x, t_sem), nan=0.0, posinf=0.0, neginf=0.0)  # (N, V, V)
        A_ljp = self.ljp_cache(labels)                              # (N, V, V)
        return A_gap, A_ljp
