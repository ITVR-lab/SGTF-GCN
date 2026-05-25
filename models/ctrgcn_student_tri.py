import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ....utils import Graph, cache_checkpoint
from ...builder import BACKBONES
from ...gcns.utils.gcn import bn_init, unit_ctrgcn
from ...gcns.utils.tcn import unit_tcn
from ...gcns.utils.msg3d_utils import MSTCN


class CTRGCNBlock_Student_Tri(nn.Module):
    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True, tcn_dropout=0):
        super().__init__()
        V = A.shape[-1]
        self.gcn = unit_ctrgcn(in_channels, out_channels, A)
        self.A_gap_s = nn.Parameter(torch.zeros(V, V))
        self.gap_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.A_ljp_s = nn.Parameter(torch.zeros(V, V))
        self.ljp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.bn_fusion = nn.BatchNorm2d(out_channels)
        self.tcn = MSTCN(out_channels, out_channels,
                         stride=stride, dilations=[1, 2],
                         residual=False, tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        y1 = self.gcn(x)
        x_gap = torch.einsum('uv,nctv->nctu', self.A_gap_s, x)
        y2 = self.gap_conv(x_gap)
        x_ljp = torch.einsum('uv,nctv->nctu', self.A_ljp_s, x)
        y3 = self.ljp_conv(x_ljp)
        y = self.relu(self.tcn(self.bn_fusion(y1 + y2 + y3)) + res)
        return y, self.gcn.A, self.A_gap_s, self.A_ljp_s

    def init_weights(self):
        self.gcn.init_weights()
        self.tcn.init_weights()


@BACKBONES.register_module()
class CTRGCNStudent_Tri(nn.Module):
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
        modules.append(CTRGCNBlock_Student_Tri(
            in_channels, ch, A.clone(), stride=1, residual=False, tcn_dropout=0))
        for i in range(2, num_stages + 1):
            in_ch = ch
            out_ch = ch * (2 if i in inflate_stages else 1)
            stride = 2 if i in down_stages else 1
            modules.append(CTRGCNBlock_Student_Tri(
                in_ch, out_ch, A.clone(), stride=stride, tcn_dropout=tcn_dropout))
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
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        all_A_adp, all_A_gap_s, all_A_ljp_s = [], [], []
        for block in self.gcn:
            x, A_adp, A_gap_s, A_ljp_s = block(x)
            all_A_adp.append(A_adp)
            all_A_gap_s.append(A_gap_s)
            all_A_ljp_s.append(A_ljp_s)
        feat = x.reshape((N, M) + x.shape[1:])
        return feat, all_A_adp, all_A_gap_s, all_A_ljp_s
