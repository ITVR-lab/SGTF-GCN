"""
SGTF-GCN Recognizer — 端到端知识蒸馏识别器。

训练：
  教师: CTRGCNTeacher (SGTF) → L_task_teacher
  学生: CTRGCNStudent         → L_task_student + λ1·L_TKD + λ2·L_KD

  L_TKD : 对每个 block、每个物理子集，用 KL 散度对齐
           softmax(A_sem/τ_TKD) 与 softmax(A_adp/τ_TKD)
  L_KD  : 对 logits 的 KL 散度（Hinton 经典 KD）

推理：仅运行学生网络，无需文本编码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from collections import OrderedDict

from ...builder import RECOGNIZERS, build_backbone, build_head


@RECOGNIZERS.register_module()
class SGTFGCNRecognizer(nn.Module):
    """SGTF-GCN 知识蒸馏识别器。

    Args:
        teacher_backbone (dict): CTRGCNTeacher 的 config。
        student_backbone (dict): CTRGCNStudent 的 config。
        cls_head         (dict): 分类头 config（教师/学生共享结构，参数独立）。
        lambda1          (float): TKD 损失权重。
        lambda2          (float): logits KD 损失权重。
        tau_TKD          (float): 拓扑 KD 温度。
        tau_KD           (float): logits KD 温度。
        train_cfg        (dict): 训练配置。
        test_cfg         (dict): 测试配置。
    """

    def __init__(self,
                 teacher_backbone,
                 student_backbone,
                 cls_head,
                 lambda1=0.5,
                 lambda2=1.0,
                 tau_TKD=2.0,
                 tau_KD=4.0,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.teacher = build_backbone(teacher_backbone)
        self.student = build_backbone(student_backbone)

        # 教师/学生各有独立参数的分类头
        self.teacher_head = build_head(cls_head)
        self.student_head = build_head(cls_head)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau_TKD = tau_TKD
        self.tau_KD = tau_KD

        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        self.init_weights()

    # ------------------------------------------------------------------
    # 语义缓存接口（训练开始前调用）
    # ------------------------------------------------------------------
    def set_ljp_cache(self, bert_embeddings: torch.Tensor):
        """将 BERT 嵌入 (K, V, C_b) 写入教师网络的 LJP cache。"""
        self.teacher.set_ljp_cache(bert_embeddings)

    def set_gap_cache(self, clip_embeddings: torch.Tensor):
        """将 CLIP 嵌入 (K, clip_dim) 存入教师网络 buffer。"""
        self.teacher.set_gap_cache(clip_embeddings)

    def set_ljp_adj_cache(self, A_ljp_all: torch.Tensor):
        """将离线 LJP 张量 (K, V, V) 写入教师各 block 的 ``LJPCache``。"""
        self.teacher.set_ljp_adj_cache(A_ljp_all)

    # ------------------------------------------------------------------
    def init_weights(self):
        self.teacher.init_weights()
        self.student.init_weights()
        self.teacher_head.init_weights()
        self.student_head.init_weights()

    # ------------------------------------------------------------------
    # 损失函数
    # ------------------------------------------------------------------
    def _tkd_loss(self,
                  all_A_sem: list,
                  all_A_adp: list) -> torch.Tensor:
        """逐 block、逐子集计算拓扑 KD 损失。

        Args:
            all_A_sem  : list[num_stages] of list[K_phy] of (N*M, V, V)
                         教师各 block 的 A_sem（已归一化）
            all_A_adp  : list[num_stages] of (K_phy, V, V)
                         学生各 block 的 learnable A_adp nn.Parameter
        Returns:
            scalar tensor
        """
        tau = self.tau_TKD
        total = 0.0
        count = 0

        for b in range(len(all_A_adp)):
            A_sem_b = all_A_sem[b]          # list[K_phy] of (N*M, V, V)
            A_adp_b = all_A_adp[b]          # (K_phy, V, V) nn.Parameter
            NM = A_sem_b[0].shape[0]

            for k in range(len(A_sem_b)):
                t_k = A_sem_b[k]                                         # (N*M, V, V)
                s_k = A_adp_b[k].unsqueeze(0).expand(NM, -1, -1)        # (N*M, V, V)

                # row-wise softmax with temperature
                P_t = F.softmax(t_k / tau, dim=-1)                      # (N*M, V, V)
                log_P_s = F.log_softmax(s_k / tau, dim=-1)              # (N*M, V, V)

                # KL(P_t || P_s)，对节点和 batch 求均值，× τ²
                kl = F.kl_div(log_P_s, P_t, reduction='batchmean')
                total = total + kl * (tau ** 2)
                count += 1

        return total / max(count, 1)

    def _logits_kd_loss(self,
                        Z_t: torch.Tensor,
                        Z_s: torch.Tensor) -> torch.Tensor:
        """Hinton 经典 KD：KL(softmax(Z_t/τ) || softmax(Z_s/τ))。"""
        tau = self.tau_KD
        P_t = F.softmax(Z_t / tau, dim=-1)
        log_P_s = F.log_softmax(Z_s / tau, dim=-1)
        return F.kl_div(log_P_s, P_t, reduction='batchmean') * (tau ** 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_train(self, keypoint, label, **kwargs):
        assert keypoint.shape[1] == 1, \
            f"Expected num_clips=1 during training, got {keypoint.shape[1]}"
        keypoint = keypoint[:, 0]          # (N, M, T, V, C)
        gt_label = label.squeeze(-1)       # (N,)

        losses = dict()

        # ---------- 教师 forward ----------
        t_feat, all_A_sem = self.teacher(keypoint, labels=gt_label)
        Z_t = self.teacher_head(t_feat)    # (N, num_classes)
        loss_t = self.teacher_head.loss(Z_t, gt_label)
        losses['loss_teacher'] = loss_t['loss_cls']

        # ---------- 学生 forward ----------
        s_feat, all_A_adp = self.student(keypoint)
        Z_s = self.student_head(s_feat)    # (N, num_classes)
        loss_s = self.student_head.loss(Z_s, gt_label)
        losses['loss_student_task'] = loss_s['loss_cls']

        # ---------- TKD 损失 ----------
        loss_tkd = self._tkd_loss(all_A_sem, all_A_adp)
        losses['loss_TKD'] = self.lambda1 * loss_tkd

        # ---------- Logits KD 损失 ----------
        loss_kd = self._logits_kd_loss(Z_t.detach(), Z_s)
        losses['loss_logits_KD'] = self.lambda2 * loss_kd

        return losses

    def forward_test(self, keypoint, **kwargs):
        """推理：仅运行学生网络。"""
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        s_feat, _ = self.student(keypoint)
        cls_score = self.student_head(s_feat)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        return self._average_clip(cls_score).data.cpu().numpy()

    def forward_eval_dual(self, keypoint, label):
        """Teacher + student probs with the same clip averaging as ``forward_test``.

        Args:
            keypoint: (N, num_clips, M, T, V, C)
            label: (N, 1) or (N,) long indices.

        Returns:
            tuple: (student_probs, teacher_probs), each (N, num_classes) on ``keypoint.device``.
        """
        bs, nc = keypoint.shape[:2]
        keypoint_nm = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        gt = label.squeeze(-1).long()
        if gt.dim() == 0:
            gt = gt.unsqueeze(0)
        labels_nm = gt.unsqueeze(1).expand(bs, nc).contiguous().reshape(-1)

        t_feat, _ = self.teacher(keypoint_nm, labels=labels_nm)
        Z_t = self.teacher_head(t_feat)
        s_feat, _ = self.student(keypoint_nm)
        Z_s = self.student_head(s_feat)

        Z_t = Z_t.reshape(bs, nc, -1)
        Z_s = Z_s.reshape(bs, nc, -1)
        mode = self.test_cfg.get('average_clips', 'prob')
        if mode == 'prob':
            p_t = F.softmax(Z_t, dim=2).mean(dim=1)
            p_s = F.softmax(Z_s, dim=2).mean(dim=1)
        elif mode == 'score':
            p_t = Z_t.mean(dim=1)
            p_s = Z_s.mean(dim=1)
        else:
            raise ValueError(f'Unknown average_clips mode: {mode}')
        return p_s, p_t

    def _average_clip(self, cls_score):
        mode = self.test_cfg.get('average_clips', 'prob')
        if mode == 'prob':
            return F.softmax(cls_score, dim=2).mean(dim=1)
        elif mode == 'score':
            return cls_score.mean(dim=1)
        raise ValueError(f'Unknown average_clips mode: {mode}')

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        if return_loss:
            assert label is not None, 'label 不能为 None'
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)

    # ------------------------------------------------------------------
    # mmcv runner 兼容接口
    # ------------------------------------------------------------------
    def parse_losses(self, losses):
        """Align with ``BaseRecognizer._parse_losses``: log_vars values become Python floats."""
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(v for k, v in log_vars.items() if 'loss' in k)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                lv = loss_value.data.clone()
                dist.all_reduce(lv.div_(dist.get_world_size()))
                log_vars[loss_name] = lv.item()
            else:
                log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self(**data_batch)
        loss, log_vars = self.parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['keypoint'])
        )
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        return self.train_step(data_batch, optimizer, **kwargs)
