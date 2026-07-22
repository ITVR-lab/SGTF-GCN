"""
SGTF-GCN (teacher-student KD) on UCF101, HRNet 2D keypoints (COCO 17, joint).

Prerequisites:
  1. python sgtfgcn_release/priors/build_semantic_cache.py \\
         --dataset ucf101 --ann_file data/ucf101/ucf101_hrnet.pkl \\
         --save_dir data/semantic_cache/ucf101 \\
         --bert_dir <local bert-base-uncased> --local_files_only

  2. Train (from repo root, mmcv + CUDA env):
       bash tools/dist_train.sh sgtfgcn_release/configs/sgtfgcn_ucf101_hrnet_j.py 1
"""

ann_file = 'data/ucf101/ucf101_hrnet.pkl'

gap_cache_path = 'data/semantic_cache/ucf101/gap_cache.pt'
ljp_cache_path = 'data/semantic_cache/ucf101/ljp_cache.pt'

semantic_cache = dict(
    gap_cache_path=gap_cache_path,
    ljp_adj_cache_path=ljp_cache_path,
)

model = dict(
    type='SGTFGCNRecognizer',
    teacher_backbone=dict(
        type='CTRGCNTeacher',
        graph_cfg=dict(layout='coco', mode='spatial'),
        clip_dim=512,
        num_classes=101,
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        sgtf_d_h=64,
        sgtf_mlp_hidden=256,
        fusion_alpha=0.5,
        fusion_beta=0.5,
        learn_fusion_scalars=True,
        num_person=2,
    ),
    student_backbone=dict(
        type='CTRGCNStudent',
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        num_person=2,
    ),
    cls_head=dict(type='GCNHead', num_classes=101, in_channels=256),
    lambda1=0.5,
    lambda2=1.0,
    tau_TKD=2.0,
    tau_KD=4.0,
)

dataset_type = 'PoseDataset'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=64),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=64, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=64, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train1'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='test1'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='test1'),
)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0004,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[35, 55],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/sgtfgcn/ucf101_hrnet_j'
