"""
Student backbone: CTR-GCN with per-block learnable A_adp.

每个 block 的 GCN 单元拥有独立的 nn.Parameter A_adp (K_phy, V, V)。
forward() 返回 (feat, all_A_adp_list)，供 Recognizer 计算 TKD 损失。
"""

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ....utils import Graph, cache_checkpoint
from ...builder import BACKBONES
from ...gcns.utils.gcn import bn_init, unit_ctrgcn
from ...gcns.utils.tcn import unit_tcn
from ...gcns.utils.msg3d_utils import MSTCN


class CTRGCNBlock_Student(nn.Module):
    """One spatio-temporal block for the student.

    Wraps unit_ctrgcn (原始实现，A_adp 是其内部 self.A nn.Parameter).
    forward() 额外返回该 block 的 learnable A_adp (K_phy, V, V)。
    """

    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True, tcn_dropout=0):
        super().__init__()
        self.gcn = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels,
                         stride=stride,
                         dilations=[1, 2],
                         residual=False,
                         tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)

    def forward(self, x):
        """
        Returns:
            y     : (N*M, C_out, T', V)
            A_adp : (K_phy, V, V)  – 当前 block 的可学习邻接参数
        """
        res = self.residual(x)
        y = self.gcn(x)
        y = self.relu(self.tcn(y) + res)
        return y, self.gcn.A   # self.gcn.A 是 unit_ctrgcn 内的 nn.Parameter

    def init_weights(self):
        self.gcn.init_weights()
        self.tcn.init_weights()


@BACKBONES.register_module()
class CTRGCNStudent(nn.Module):
    """Lightweight student backbone (standard CTR-GCN).

    每个 block 独立维护一套 A_adp，forward 返回全部 block 的 A_adp 供 TKD。

    Args:
        graph_cfg     : 传给 Graph()。
        in_channels   : 坐标通道数（3dkp=3）。
        base_channels : 64。
        num_stages    : 10。
        inflate_stages: (5, 8)。
        down_stages   : (5, 8)。
        num_person    : 2。
        pretrained    : 可选 checkpoint 路径。
        tcn_dropout   : TCN dropout。
    """

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 num_person=2,
                 pretrained=None,
                 tcn_dropout=0,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        V = A.shape[-1]
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * V)

        inflate_stages = list(inflate_stages)
        down_stages = list(down_stages)

        modules = []
        ch = base_channels

        # block 1
        modules.append(CTRGCNBlock_Student(
            in_channels, ch, A.clone(),
            stride=1, residual=False, tcn_dropout=0))

        for i in range(2, num_stages + 1):
            in_ch = ch
            out_ch = ch * (2 if i in inflate_stages else 1)
            stride = 2 if i in down_stages else 1
            modules.append(CTRGCNBlock_Student(
                in_ch, out_ch, A.clone(),
                stride=stride, tcn_dropout=tcn_dropout))
            ch = out_ch

        self.gcn = nn.ModuleList(modules)
        self.num_stages = num_stages
        self.pretrained = pretrained

    def init_weights(self):
        bn_init(self.data_bn, 1)
        for block in self.gcn:
            block.init_weights()
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        """
        Args:
            x: (N, M, T, V, C)
        Returns:
            feat       : (N, M, C_out, T', V)
            all_A_adp  : list[num_stages]，每项是 (K_phy, V, V) nn.Parameter
        """
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        all_A_adp = []
        for block in self.gcn:
            x, A_adp = block(x)
            all_A_adp.append(A_adp)   # (K_phy, V, V)

        feat = x.reshape((N, M) + x.shape[1:])
        return feat, all_A_adp
