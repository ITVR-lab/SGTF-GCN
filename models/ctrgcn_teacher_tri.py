"""
Teacher backbone for TriSGTFGCN.
"""

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ....utils import Graph, cache_checkpoint
from ...builder import BACKBONES
from ...gcns.utils.gcn import bn_init
from ...gcns.utils.tcn import unit_tcn
from ...gcns.utils.msg3d_utils import MSTCN
from .sgtf_module_v2 import SGTFModuleV2
from .unit_ctrgcn_teacher_tri import unit_ctrgcn_teacher_tri


class CTRGCNBlock_Teacher_Tri(nn.Module):
    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True,
                 kernel_size=5, dilations=(1, 2), tcn_dropout=0):
        super().__init__()
        self.gcn = unit_ctrgcn_teacher_tri(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         dilations=list(dilations),
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

    def forward(self, x, A_gap, A_ljp):
        res = self.residual(x)
        x, A_gap, A_ljp = self.gcn(x, A_gap, A_ljp)
        x = self.relu(self.tcn(x) + res)
        return x, A_gap, A_ljp

    def init_weights(self):
        self.gcn.init_weights()
        self.tcn.init_weights()


@BACKBONES.register_module()
class CTRGCNTeacher_Tri(nn.Module):
    def __init__(self,
                 graph_cfg,
                 clip_dim=512,
                 num_classes=60,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 sgtf_d_h=64,
                 sgtf_mlp_hidden=256,
                 num_person=2,
                 pretrained=None,
                 tcn_dropout=0,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        V = A.shape[-1]
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * V)

        inflate_stages = list(inflate_stages)
        down_stages = list(down_stages)

        # Compute per-block output channels
        block_out_channels = [base_channels]
        ch = base_channels
        for i in range(2, num_stages + 1):
            out_ch = ch * (2 if i in inflate_stages else 1)
            block_out_channels.append(out_ch)
            ch = out_ch

        # Build blocks
        modules = []
        ch = base_channels
        modules.append(CTRGCNBlock_Teacher_Tri(
            in_channels, ch, A.clone(), stride=1, residual=False, tcn_dropout=0))
        for i in range(2, num_stages + 1):
            in_ch = ch
            out_ch = ch * (2 if i in inflate_stages else 1)
            stride = 2 if i in down_stages else 1
            modules.append(CTRGCNBlock_Teacher_Tri(
                in_ch, out_ch, A.clone(), stride=stride, tcn_dropout=tcn_dropout))
            ch = out_ch

        self.gcn = nn.ModuleList(modules)
        self.num_stages = num_stages

        # SGTFModuleV2 per block
        self.sgtf_list = nn.ModuleList([
            SGTFModuleV2(
                clip_dim=clip_dim,
                feat_channels=in_channels if i == 0 else block_out_channels[i - 1],
                num_classes=num_classes,
                num_joints=V,
                d_h=sgtf_d_h,
                mlp_hidden=sgtf_mlp_hidden,
            )
            for i in range(num_stages)
        ])

        self.register_buffer('gap_cache', torch.zeros(num_classes, clip_dim), persistent=True)
        self.pretrained = pretrained

    def set_gap_cache(self, clip_embeddings):
        emb = clip_embeddings.to(device=self.gap_cache.device, dtype=self.gap_cache.dtype)
        if emb.shape != self.gap_cache.shape:
            raise ValueError('gap_cache shape mismatch: {} vs {}'.format(
                tuple(emb.shape), tuple(self.gap_cache.shape)))
        self.gap_cache.copy_(emb)

    def set_ljp_adj_cache(self, A_ljp_all):
        for sgtf in self.sgtf_list:
            dst = sgtf.ljp_cache.cache
            if A_ljp_all.shape != dst.shape:
                raise ValueError('LJP cache shape mismatch: {} vs {}'.format(
                    tuple(A_ljp_all.shape), tuple(dst.shape)))
            dst.copy_(A_ljp_all.to(device=dst.device, dtype=dst.dtype))

    def init_weights(self):
        bn_init(self.data_bn, 1)
        for block in self.gcn:
            block.init_weights()
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, labels=None):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        NM = N * M

        if labels is not None:
            t_sem_nm = self.gap_cache[labels].repeat_interleave(M, dim=0)
            labels_nm = labels.repeat_interleave(M, dim=0)
        else:
            t_sem_nm = None
            labels_nm = None

        all_A_gap, all_A_ljp = [], []
        for i, block in enumerate(self.gcn):
            if t_sem_nm is not None:
                A_gap, A_ljp = self.sgtf_list[i](x, t_sem_nm, labels_nm)
            else:
                A_gap = torch.zeros(NM, V, V, device=x.device, dtype=x.dtype)
                A_ljp = torch.zeros(NM, V, V, device=x.device, dtype=x.dtype)
            x, A_gap, A_ljp = block(x, A_gap, A_ljp)
            all_A_gap.append(A_gap)
            all_A_ljp.append(A_ljp)

        feat = x.reshape((N, M) + x.shape[1:])
        return feat, all_A_gap, all_A_ljp
