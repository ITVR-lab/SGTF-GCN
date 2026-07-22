"""
Tri-stream teacher GCN unit for TriSGTFGCN.

Three parallel streams per block:
  Stream 1 — Original CTR-GCN:  CTRGC(x, A_phy[k]) * num_subsets  (fully preserved)
  Stream 2 — GAP stream:         A_gap @ x -> gap_conv             (semantic class similarity)
  Stream 3 — LJP stream:         A_ljp @ x -> ljp_conv             (joint function prior)

Output = BN(y1 + y2 + y3) + residual  (1:1:1 fusion)
Returns (y, A_gap, A_ljp) so the recognizer can collect topologies for TKD loss.
"""

import torch
import torch.nn as nn

from ...gcns.utils.gcn import conv_init, bn_init, CTRGC


class unit_ctrgcn_teacher_tri(nn.Module):
    """Tri-stream teacher GCN unit.

    Args:
        in_channels  : input feature channels C_in
        out_channels : output feature channels C_out
        A            : (K_phy, V, V) physical adjacency (buffer, not trained)
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.num_subset = A.shape[0]

        # ---- Stream 1: Original CTR-GCN (one CTRGC per physical subset) ----
        # CTRGC.forward(x, A, alpha): A is (V, V), alpha is scalar
        self.convs = nn.ModuleList([
            CTRGC(in_channels, out_channels)
            for _ in range(self.num_subset)
        ])
        self.register_buffer('A_phy', A.clone())   # (K_phy, V, V), non-trainable
        self.alpha = nn.Parameter(torch.zeros(1))  # dynamic term weight for stream1

        # ---- Stream 2: GAP stream ----
        self.gap_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ---- Stream 3: LJP stream ----
        self.ljp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ---- Shared output BN + residual ----
        self.down = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                          nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else (lambda x: x)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor,
                A_gap: torch.Tensor,
                A_ljp: torch.Tensor):
        """
        Args:
            x:     (N, C_in, T, V)
            A_gap: (N, V, V)  per-sample GAP adjacency from SGTFModuleV2
            A_ljp: (N, V, V)  per-sample LJP adjacency from SGTFModuleV2
        Returns:
            y:     (N, C_out, T, V)
            A_gap: pass-through for TKD loss collection
            A_ljp: pass-through for TKD loss collection
        """
        # --- Stream 1: Original CTR-GCN ---
        # CTRGC expects A as (V,V); it internally does A[None,None] to broadcast
        y1 = None
        for i, conv in enumerate(self.convs):
            z = conv(x, self.A_phy[i], self.alpha)  # (N, C_out, T, V)
            y1 = z if y1 is None else y1 + z

        # --- Stream 2: GAP aggregation then conv ---
        # A_gap: (N,V,V); x: (N,C,T,V)
        # x_gap[n,c,t,u] = sum_v A_gap[n,u,v] * x[n,c,t,v]
        x_gap = torch.einsum('nuv,nctv->nctu', A_gap, x)   # (N, C, T, V)
        y2 = self.gap_conv(x_gap)

        # --- Stream 3: LJP aggregation then conv ---
        x_ljp = torch.einsum('nuv,nctv->nctu', A_ljp, x)   # (N, C, T, V)
        y3 = self.ljp_conv(x_ljp)

        y = self.bn(y1 + y2 + y3) + self.down(x)
        return self.relu(y), A_gap, A_ljp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    conv_init(m)
                else:
                    import torch.nn.init as init
                    init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
