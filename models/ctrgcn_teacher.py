"""
Teacher backbone: SGTF-CTR-GCN.

10 个 block，每个 block 的空间 GCN 单元接受来自 SGTF 的 A_sem_list。
forward() 返回 (feat, all_A_sem) 供 TKD 损失使用。
"""

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ....utils import Graph, cache_checkpoint
from ...builder import BACKBONES
from ...gcns.utils.gcn import bn_init
from ...gcns.utils.tcn import unit_tcn
from ...gcns.utils.msg3d_utils import MSTCN
from .sgtf_module import SGTFModule
from .unit_ctrgcn_teacher import unit_ctrgcn_teacher


class CTRGCNBlock_Teacher(nn.Module):
    """One spatio-temporal block for the teacher network."""

    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True,
                 kernel_size=5, dilations=(1, 2), tcn_dropout=0):
        super().__init__()
        self.gcn = unit_ctrgcn_teacher(in_channels, out_channels, A)
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

    def forward(self, x, A_sem_list):
        res = self.residual(x)
        x, A_sem_list = self.gcn(x, A_sem_list)
        x = self.relu(self.tcn(x) + res)
        return x, A_sem_list

    def init_weights(self):
        self.gcn.init_weights()
        self.tcn.init_weights()


@BACKBONES.register_module()
class CTRGCNTeacher(nn.Module):
    """SGTF-CTR-GCN 教师 backbone。

    Args:
        graph_cfg       : 传给 Graph()。
        clip_dim        : CLIP 文本嵌入维度（ViT-B/32 = 512）。
        num_classes     : 动作类别数 K。
        in_channels     : 输入坐标通道数（3dkp=3）。
        base_channels   : 64。
        num_stages      : 10。
        inflate_stages  : 通道翻倍的 block 序号（1-indexed）。
        down_stages     : 时序下采样的 block 序号（1-indexed）。
        sgtf_d_h        : GAP 注意力投影维度。
        sgtf_mlp_hidden : GAP MLP 隐层维度。
        fusion_alpha    : GAP 分支可学习权重的初值。
        fusion_beta     : LJP 分支可学习权重的初值。
        learn_fusion_scalars: 是否训练 ``alpha``/``beta``（默认 True，与论文设置一致）。
        num_person      : 每样本人数（用于 data_bn）。
        pretrained      : 可选 checkpoint 路径。
        tcn_dropout     : TCN dropout。
    """

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
        fusion_alpha=0.5,
        fusion_beta=0.5,
        learn_fusion_scalars=True,
        num_person=2,
        pretrained=None,
        tcn_dropout=0,
        **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)   # (K_phy, V, V)

        V = A.shape[-1]
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * V)

        inflate_stages = list(inflate_stages)
        down_stages = list(down_stages)

        # ---- 计算每个 block 的输出通道，用于构建对应的 SGTF ----
        block_out_channels = []
        ch = base_channels
        for i in range(1, num_stages + 1):
            if i in inflate_stages:
                ch *= 2
            block_out_channels.append(ch)
        # block_out_channels[0] 对应 block1 的输出通道（不 inflate）
        # 重新按实际构建逻辑推算
        block_out_channels = []
        ch = base_channels
        # block 1: in_channels -> base_channels, no inflate
        block_out_channels.append(base_channels)
        for i in range(2, num_stages + 1):
            out_ch = ch * (2 if i in inflate_stages else 1)
            block_out_channels.append(out_ch)
            ch = out_ch

        # ---- 构建 10 个 block ----
        modules = []
        ch = base_channels
        modules.append(CTRGCNBlock_Teacher(
            in_channels, ch, A.clone(),
            stride=1, residual=False, tcn_dropout=0))

        for i in range(2, num_stages + 1):
            in_ch = ch
            out_ch = ch * (2 if i in inflate_stages else 1)
            stride = 2 if i in down_stages else 1
            modules.append(CTRGCNBlock_Teacher(
                in_ch, out_ch, A.clone(),
                stride=stride, tcn_dropout=tcn_dropout))
            ch = out_ch

        self.gcn = nn.ModuleList(modules)
        self.num_stages = num_stages

        # ---- 每个 block 前一套 SGTF：GAP 的 feat 维必须对该 block 的 *输入* 通道数 ----
        # sgtf_list[i] 看到的是进入第 i 个 block 之前的 x，故 i==0 时为 in_channels（如 3），
        # i>=1 时为上一 block 输出 block_out_channels[i - 1]（此前误用 block 输出导致 C=64 与 C=3 冲突）。
        self.sgtf_list = nn.ModuleList([
            SGTFModule(
                A_phy=A.clone(),
                clip_dim=clip_dim,
                feat_channels=in_channels if i == 0 else block_out_channels[i - 1],
                num_classes=num_classes,
                num_joints=V,
                d_h=sgtf_d_h,
                mlp_hidden=sgtf_mlp_hidden,
                fusion_alpha=fusion_alpha,
                fusion_beta=fusion_beta,
                learn_fusion_scalars=learn_fusion_scalars,
            )
            for i in range(num_stages)
        ])

        # Persist in ``state_dict`` so checkpoints reload cleanly (``set_gap_cache`` copies in).
        self.register_buffer(
            'gap_cache',
            torch.zeros(num_classes, clip_dim),
            persistent=True)

        self.pretrained = pretrained

    # ------------------------------------------------------------------
    # 语义缓存接口（训练前调用一次）
    # ------------------------------------------------------------------
    def set_ljp_cache(self, bert_embeddings: torch.Tensor):
        """将 (K, V, C_b) BERT 嵌入写入每个 block 的 LJP cache。"""
        for sgtf in self.sgtf_list:
            sgtf.ljp_cache.set_cache(bert_embeddings.to(
                next(self.parameters()).device))

    def set_gap_cache(self, clip_embeddings: torch.Tensor):
        """Copy (K, clip_dim) CLIP embeddings into ``gap_cache`` (indexed by label in forward)."""
        emb = clip_embeddings.to(device=self.gap_cache.device, dtype=self.gap_cache.dtype)
        if emb.shape != self.gap_cache.shape:
            raise ValueError(
                f'gap_cache expects shape {tuple(self.gap_cache.shape)}, got {tuple(emb.shape)}')
        self.gap_cache.copy_(emb)

    def set_ljp_adj_cache(self, A_ljp_all: torch.Tensor):
        """写入离线生成的 LJP 张量 (K, V, V)，与 ``build_semantic_cache.py`` 产物一致。"""
        for sgtf in self.sgtf_list:
            dst = sgtf.ljp_cache.cache
            if A_ljp_all.shape != dst.shape:
                raise ValueError(
                    f'LJP cache shape {tuple(A_ljp_all.shape)} != expected '
                    f'{tuple(dst.shape)} (check num_classes / num_joints / dataset)')
            dst.copy_(A_ljp_all.to(device=dst.device, dtype=dst.dtype))

    # ------------------------------------------------------------------
    def init_weights(self):
        bn_init(self.data_bn, 1)
        for block in self.gcn:
            block.init_weights()
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, labels=None):
        """
        Args:
            x      : (N, M, T, V, C)
            labels : (N,) 训练时的 GT 类别索引，验证/测试时为 None
        Returns:
            feat       : (N, M, C_out, T', V)
            all_A_sem  : list[num_stages]，每项是 list[K_phy] of (N*M, V, V)
        """
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        NM = N * M

        # ---- 准备 CLIP 嵌入 & labels（N*M 维度）----
        if labels is not None and hasattr(self, 'gap_cache'):
            t_sem_nm = self.gap_cache[labels].repeat_interleave(M, dim=0)  # (N*M, clip_dim)
            labels_nm = labels.repeat_interleave(M, dim=0)                 # (N*M,)
        else:
            t_sem_nm = None
            labels_nm = None

        all_A_sem = []
        for i, block in enumerate(self.gcn):
            if t_sem_nm is not None:
                A_sem_list = self.sgtf_list[i](x, t_sem_nm, labels_nm)
            else:
                # fallback: 使用 A_phy 广播，不做语义融合
                A_sem_list = [
                    self.sgtf_list[i].A_phy[k].unsqueeze(0).expand(NM, -1, -1)
                    for k in range(self.sgtf_list[i].num_subsets)
                ]
            x, A_sem_list = block(x, A_sem_list)
            all_A_sem.append(A_sem_list)   # list of K_phy tensors (N*M, V, V)

        feat = x.reshape((N, M) + x.shape[1:])
        return feat, all_A_sem
