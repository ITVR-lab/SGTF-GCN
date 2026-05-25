import torch
import torch.nn as nn

from ...utils import Graph
from ..builder import BACKBONES
from ..recognizers.recognizergcn import RecognizerGCN
from ..utils import unit_ctrgcn

# Lightweight knowledge-distillation wrapper for CTR-GCN
# - keeps original A as Aphy
# - introduces Asem as an auxiliary semantic adjacency (to be provided via graph_cfg_sem)


class CTRGCN_KD(nn.Module):
    def __init__(self, graph_cfg, graph_cfg_sem=None, in_channels=3, base_channels=64, num_stages=10, pretrained=None, **kwargs):
        super().__init__()
        # physical graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('Aphy', A)

        # semantic graph (optional)
        if graph_cfg_sem is not None:
            self.graph_sem = Graph(**graph_cfg_sem)
            Asem = torch.tensor(self.graph_sem.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('Asem_init', Asem)
        else:
            self.graph_sem = None
            self.register_buffer('Asem_init', torch.zeros_like(A))

        # backbone will be a RecognizerGCN with CTRGCN backbone; we keep it simple and reuse it
        backbone_cfg = dict(type='CTRGCN', graph_cfg=graph_cfg, in_channels=in_channels, base_channels=base_channels, num_stages=num_stages, pretrained=pretrained)
        self.teacher = RecognizerGCN(backbone_cfg, cls_head=None)
        # student can be a smaller backbone; for now reuse same class, user can pass smaller base_channels later
        self.student = RecognizerGCN(backbone_cfg, cls_head=None)

    def forward(self, x):
        # return teacher and student features (logits handled by heads externally)
        t_feat = self.teacher(x)
        s_feat = self.student(x)
        return t_feat, s_feat
