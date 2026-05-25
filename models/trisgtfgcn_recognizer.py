import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict

from ...builder import RECOGNIZERS, build_backbone, build_head


@RECOGNIZERS.register_module()
class TriSGTFGCNRecognizer(nn.Module):
    def __init__(self,
                 teacher_backbone,
                 student_backbone,
                 cls_head,
                 lambda_gap=0.5,
                 lambda_ljp=0.3,
                 lambda2=1.0,
                 tau_TKD=2.0,
                 tau_KD=4.0,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.teacher = build_backbone(teacher_backbone)
        self.student = build_backbone(student_backbone)
        self.teacher_head = build_head(cls_head)
        self.student_head = build_head(cls_head)
        self.lambda_gap = lambda_gap
        self.lambda_ljp = lambda_ljp
        self.lambda2 = lambda2
        self.tau_TKD = tau_TKD
        self.tau_KD = tau_KD
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.init_weights()

    def set_gap_cache(self, clip_embeddings):
        self.teacher.set_gap_cache(clip_embeddings)

    def set_ljp_adj_cache(self, A_ljp_all):
        self.teacher.set_ljp_adj_cache(A_ljp_all)

    def init_weights(self):
        self.teacher.init_weights()
        self.student.init_weights()
        self.teacher_head.init_weights()
        self.student_head.init_weights()

    def _tkd_loss_single(self, all_A_teacher, all_A_student):
        tau = self.tau_TKD
        total = 0.0
        count = 0
        for b in range(len(all_A_teacher)):
            t_A = all_A_teacher[b]   # (N*M, V, V)
            s_A = all_A_student[b]   # (V, V)
            NM = t_A.shape[0]
            V = t_A.shape[-1]
            s_A_exp = s_A.unsqueeze(0).expand(NM, -1, -1)
            P_t = F.softmax(t_A / tau, dim=-1)
            log_P_s = F.log_softmax(s_A_exp / tau, dim=-1)
            kl = F.kl_div(
                log_P_s.reshape(-1, V),
                P_t.reshape(-1, V),
                reduction='batchmean'
            ) * (tau ** 2)
            total = total + kl
            count += 1
        return total / max(count, 1)

    def _logits_kd_loss(self, Z_t, Z_s):
        tau = self.tau_KD
        P_t = F.softmax(Z_t / tau, dim=-1)
        log_P_s = F.log_softmax(Z_s / tau, dim=-1)
        return F.kl_div(log_P_s, P_t, reduction='batchmean') * (tau ** 2)

    def forward_train(self, keypoint, label, **kwargs):
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]
        gt_label = label.squeeze(-1)
        losses = dict()
        t_feat, all_A_gap_t, all_A_ljp_t = self.teacher(keypoint, labels=gt_label)
        Z_t = self.teacher_head(t_feat)
        loss_t = self.teacher_head.loss(Z_t, gt_label)
        losses['loss_teacher'] = loss_t['loss_cls']
        s_feat, all_A_adp, all_A_gap_s, all_A_ljp_s = self.student(keypoint)
        Z_s = self.student_head(s_feat)
        loss_s = self.student_head.loss(Z_s, gt_label)
        losses['loss_student_task'] = loss_s['loss_cls']
        losses['loss_TKD_gap'] = self.lambda_gap * self._tkd_loss_single(all_A_gap_t, all_A_gap_s)
        losses['loss_TKD_ljp'] = self.lambda_ljp * self._tkd_loss_single(all_A_ljp_t, all_A_ljp_s)
        losses['loss_logits_KD'] = self.lambda2 * self._logits_kd_loss(Z_t.detach(), Z_s)
        return losses

    def forward_test(self, keypoint, **kwargs):
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        s_feat, _, _, _ = self.student(keypoint)
        cls_score = self.student_head(s_feat)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        return self._average_clip(cls_score).data.cpu().numpy()

    def _average_clip(self, cls_score):
        mode = self.test_cfg.get('average_clips', 'prob')
        if mode == 'prob':
            return F.softmax(cls_score, dim=2).mean(dim=1)
        elif mode == 'score':
            return cls_score.mean(dim=1)
        raise ValueError('Unknown average_clips mode: {}'.format(mode))

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        if return_loss:
            assert label is not None
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)

    def parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError('{} is not a tensor or list'.format(loss_name))
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
        return dict(loss=loss, log_vars=log_vars,
                    num_samples=len(data_batch['keypoint']))

    def val_step(self, data_batch, optimizer, **kwargs):
        return self.train_step(data_batch, optimizer, **kwargs)
