# Copyright (c) OpenMMLab. All rights reserved.
"""Load offline GAP / LJP tensors into ``SGTFGCNRecognizer`` before training."""

import os
import os.path as osp

import torch


def resolve_checkpoint_path(path, cfg):
    """Resolve possibly-relative paths against cwd then repo root (next to configs/)."""
    if osp.isabs(path) and osp.isfile(path):
        return path
    if not osp.isabs(path):
        c1 = osp.abspath(osp.join(os.getcwd(), path))
        if osp.isfile(c1):
            return c1
        fname = getattr(cfg, 'filename', None) or getattr(cfg, '_filename', None)
        if fname:
            cfg_dir = osp.dirname(osp.abspath(fname))
            repo_root = osp.abspath(osp.join(cfg_dir, '..', '..'))
            c2 = osp.join(repo_root, path)
            if osp.isfile(c2):
                return c2
    raise FileNotFoundError(f'Semantic cache file not found: {path}')


def load_semantic_cache(model, sem_cfg, cfg):
    """Load ``gap_cache.pt`` / ``ljp_cache.pt`` into recognizer (CPU tensors).

    Args:
        model (nn.Module): Built ``SGTFGCNRecognizer`` (before ``.cuda()``).
        sem_cfg (dict): Must contain ``gap_cache_path`` and ``ljp_adj_cache_path``.
        cfg (mmcv.Config): Used only for path resolution.
    """
    gap_path = resolve_checkpoint_path(sem_cfg['gap_cache_path'], cfg)
    ljp_path = resolve_checkpoint_path(sem_cfg['ljp_adj_cache_path'], cfg)

    gap = torch.load(gap_path, map_location='cpu')
    ljp = torch.load(ljp_path, map_location='cpu')

    if hasattr(model, 'set_gap_cache'):
        model.set_gap_cache(gap)
    else:
        raise TypeError(f'Model {type(model).__name__} has no set_gap_cache')

    if hasattr(model, 'set_ljp_adj_cache'):
        model.set_ljp_adj_cache(ljp)
    else:
        raise TypeError(f'Model {type(model).__name__} has no set_ljp_adj_cache')

