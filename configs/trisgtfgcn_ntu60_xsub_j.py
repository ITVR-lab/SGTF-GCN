gap_cache_path = 'sgtfgcn_release/priors/cache/gap_cache.pt'
ljp_cache_path = 'sgtfgcn_release/priors/cache/ljp_cache.pt'

semantic_cache = dict(
    gap_cache_path=gap_cache_path,
    ljp_adj_cache_path=ljp_cache_path,
)

model = dict(
    type='TriSGTFGCNRecognizer',
    teacher_backbone=dict(
        type='CTRGCNTeacher_Tri',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        clip_dim=512,
        num_classes=60,
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        sgtf_d_h=64,
        sgtf_mlp_hidden=256,
        num_person=2,
    ),
    student_backbone=dict(
        type='CTRGCNStudent_Tri',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        num_person=2,
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=60,
        in_channels=256,
    ),
    lambda_gap=0.5,
    lambda_ljp=0.3,
    lambda2=1.0,
    tau_TKD=2.0,
    tau_KD=4.0,
)

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=64),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=64, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
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
    train=dict(type=dataset_type, ann_file=ann_file,
               pipeline=train_pipeline, split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file,
             pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file,
              pipeline=test_pipeline, split='xsub_val'),
)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0004, nesterov=True)
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
work_dir = './work_dirs/sgtfgcn/ntu60_xsub_j_trisgtfgcn'
