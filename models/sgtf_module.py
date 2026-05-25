"""
Semantics-Guided Topology Fusion (SGTF) Module.

Implements:
  - GAP (Global Action-context Prior): CLIP-encoded action descriptions
    modulate inter-joint attention to yield A_gap (N, V, V).
  - LJP (Local Joint-relation Prior): per-class BERT-encoded joint-function
    templates, precomputed offline, yielding A_ljp (V, V) per class.
  - Topology Fusion:
      A_sem = D^{-1/2} (A_phy + alpha * A_gap + beta * A_ljp) D^{-1/2}

Only the teacher network uses SGTF at training time.
During inference the student runs with its own learnable A_adp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAPModule(nn.Module):
    """Global Action-context Prior (GAP).

    Uses pre-computed CLIP text embeddings (one per action class, stored in a
    cache tensor) together with the skeleton features to build A_gap.

    Args:
        clip_dim (int): Dimension of the CLIP text embedding (512 for ViT-B/32).
        feat_channels (int): Spatial feature channels C fed into this module.
        d_h (int): Projection dimension for the attention inner product.
        mlp_hidden (int): Hidden dim of the MLP that maps clip->S_c.
    """

    def __init__(self, clip_dim: int, feat_channels: int, d_h: int = 64,
                 mlp_hidden: int = 256):
        super().__init__()
        self.d_h = d_h
        # MLP: clip_dim -> mlp_hidden -> feat_channels  (produces S_c)
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, feat_channels),
        )
        # Query / Key projections  (C -> d_h)
        self.W_Q = nn.Linear(feat_channels, d_h, bias=False)
        self.W_K = nn.Linear(feat_channels, d_h, bias=False)

    def forward(self, x: torch.Tensor, t_sem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N, C, T, V)  skeleton feature tensor from the current block
            t_sem: (N, clip_dim) pre-retrieved CLIP text embedding for each sample
        Returns:
            A_gap: (N, V, V) row-wise softmax normalised attention adjacency
        """
        N, C, T, V = x.shape

        # --- temporal average to get compact spatial representation ---
        x_bar = x.mean(dim=2)                          # (N, C, V)
        x_bar = x_bar.permute(0, 2, 1).contiguous()    # (N, V, C)

        # --- channel-wise semantic modulation vector S_c ---
        S_c = self.mlp(t_sem)                           # (N, C)
        # broadcast S_c to all joints: (N, 1, C) * (N, V, C) -> (N, V, C)
        x_tilde = x_bar * S_c.unsqueeze(1)              # (N, V, C)

        # --- attention: Q from x_bar, K from semantically modulated x_tilde ---
        Q = self.W_Q(x_bar)                             # (N, V, d_h)
        E = self.W_K(x_tilde)                           # (N, V, d_h)

        # scaled dot-product -> row-wise softmax
        attn = torch.bmm(Q, E.transpose(1, 2)) / (self.d_h ** 0.5)  # (N, V, V)
        A_gap = F.softmax(attn, dim=-1)                 # (N, V, V)
        return A_gap


class LJPCache(nn.Module):
    """Local Joint-relation Prior (LJP) cache.

    Stores K pre-computed (V, V) cosine-similarity adjacency matrices built
    from BERT joint-function embeddings.  At training time only a label-indexed
    lookup is needed (O(1)).

    The cache is registered as a non-trainable buffer so it is saved with the
    model checkpoint and moved to the correct device automatically.

    Args:
        num_classes (int): K – number of action classes.
        num_joints  (int): V – number of skeleton joints.
    """

    def __init__(self, num_classes: int, num_joints: int):
        super().__init__()
        # (K, V, V) – filled externally before training via set_cache()
        self.register_buffer(
            'cache',
            torch.zeros(num_classes, num_joints, num_joints)
        )
        self.num_classes = num_classes
        self.num_joints = num_joints

    @torch.no_grad()
    def set_cache(self, bert_embeddings: torch.Tensor):
        """Build and store the LJP cache from BERT embeddings.

        Args:
            bert_embeddings: (K, V, C_b) – BERT Pooler outputs for every
                             (class, joint) pair.  C_b = 768 for BERT-base.
        """
        K, V, Cb = bert_embeddings.shape
        assert K == self.num_classes and V == self.num_joints, \
            f"Expected ({self.num_classes}, {self.num_joints}, *), got {bert_embeddings.shape}"

        # L2-normalise each embedding
        F_hat = F.normalize(bert_embeddings, p=2, dim=-1)  # (K, V, C_b)

        # Gram matrix: cosine similarity between all joint pairs per class
        # (K, V, V)
        A_ljp_all = torch.bmm(F_hat, F_hat.transpose(1, 2))
        self.cache.copy_(A_ljp_all)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """O(1) lookup.

        Args:
            labels: (N,) long tensor of ground-truth class indices.
        Returns:
            A_ljp: (N, V, V)
        """
        return self.cache[labels]   # fancy indexing, shape (N, V, V)


class SGTFModule(nn.Module):
    """Semantics-Guided Topology Fusion module (teacher only).

    Fuses:
        A_sem = D^{-1/2} (A_phy + alpha * A_gap + beta * A_ljp) D^{-1/2}

    Args:
        A_phy        : (K_phy, V, V) physical adjacency (K_phy subsets).
        clip_dim     : CLIP text embedding dimension (default 512).
        feat_channels: C of the skeleton features at the layer where SGTF is applied.
        num_classes  : K action classes.
        num_joints   : V joints.
        d_h          : Attention projection dimension.
        mlp_hidden   : Hidden dim in the GAP MLP.
        fusion_alpha : Weight on ``A_gap``. If ``learn_fusion_scalars=False`` (default),
                       stored as a fixed buffer (no grad). Otherwise initial value for
                       ``nn.Parameter``.
        fusion_beta  : Weight on ``A_ljp`` (same semantics as ``fusion_alpha``).
        learn_fusion_scalars: If True, ``alpha``/``beta`` are trainable ``nn.Parameter``.
    """

    def __init__(self,
                 A_phy: torch.Tensor,
                 clip_dim: int,
                 feat_channels: int,
                 num_classes: int,
                 num_joints: int,
                 d_h: int = 64,
                 mlp_hidden: int = 256,
                 fusion_alpha: float = 0.5,
                 fusion_beta: float = 0.5,
                 learn_fusion_scalars: bool = False):
        super().__init__()

        # register Aphy as non-trainable buffer (one tensor per subset)
        self.register_buffer('A_phy', A_phy.clone())  # (K_phy, V, V)
        self.num_subsets = A_phy.shape[0]

        self.gap = GAPModule(clip_dim, feat_channels, d_h, mlp_hidden)
        self.ljp_cache = LJPCache(num_classes, num_joints)

        # A_hat = A_phy + alpha*A_gap + beta*A_ljp  (default: fixed scalars, stable at cold start)
        a = torch.tensor([fusion_alpha], dtype=torch.float32)
        b = torch.tensor([fusion_beta], dtype=torch.float32)
        if learn_fusion_scalars:
            self.alpha = nn.Parameter(a.clone())
            self.beta = nn.Parameter(b.clone())
        else:
            self.register_buffer('alpha', a)
            self.register_buffer('beta', b)

    def _degree_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """Symmetric degree normalisation: D^{-1/2} A D^{-1/2}.

        Args:
            A: (N, V, V)
        Returns:
            A_norm: (N, V, V)
        """
        # degree = sum of |A_ij| per row
        D = A.abs().sum(dim=-1)                        # (N, V)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0   # handle zero-degree nodes
        # (N, V, V)
        A_norm = D_inv_sqrt.unsqueeze(-1) * A * D_inv_sqrt.unsqueeze(-2)
        return A_norm

    def forward(self,
                x: torch.Tensor,
                t_sem: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (N, C, T, V)
            t_sem:  (N, clip_dim) CLIP embeddings for this batch (retrieved from cache)
            labels: (N,) ground-truth class indices (long)
        Returns:
            A_sem_list: list of K_phy tensors, each (N, V, V)  – one per physical subset.
                        Each tensor is the fused & normalised adjacency for that subset.
        """
        N = x.shape[0]
        V = self.A_phy.shape[-1]

        A_gap = torch.nan_to_num(
            self.gap(x, t_sem), nan=0.0, posinf=0.0, neginf=0.0)   # (N, V, V)
        A_ljp = self.ljp_cache(labels)                          # (N, V, V)

        A_sem_list = []
        for k in range(self.num_subsets):
            A_k_phy = self.A_phy[k].unsqueeze(0).expand(N, -1, -1)  # (N, V, V)
            A_hat = A_k_phy + self.alpha * A_gap + self.beta * A_ljp
            A_sem_list.append(self._degree_normalize(A_hat))    # (N, V, V)

        return A_sem_list
