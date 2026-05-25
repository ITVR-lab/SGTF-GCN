"""
Training config for SGTF-GCN on NTU RGB+D 60 X-Sub (joint stream).

Two-step usage:
  1. Build caches (once). Offline BERT (no HuggingFace download):
       python configs/sgtfgcn/build_semantic_cache.py \
           --dataset nturgb+d --save_dir data/semantic_cache/ntu60 \
           --bert_dir data/hf_models/bert-base-uncased --local_files_only

     (Paths are relative to repo root ``pyskl-main``. Adjust ``--bert_dir`` if your
     snapshot lives elsewhere.)

  2. Train:
       bash tools/dist_train.sh configs/sgtfgcn/sgtfgcn_ntu60_xsub_j.py 4

The recognizer (SGTFGCNRecognizer) loads the caches at runtime via
a custom hook (or you can call model.set_gap_cache / model.set_ljp_cache
directly in your training script).
"""

# -----------------------------------------------------------------------
# Paths to pre-computed semantic caches (build with build_semantic_cache.py)
# -----------------------------------------------------------------------
gap_cache_path = 'data/semantic_cache/ntu60/gap_cache.pt'
ljp_cache_path = 'data/semantic_cache/ntu60/ljp_cache.pt'

semantic_cache = dict(
    gap_cache_path=gap_cache_path,
    ljp_adj_cache_path=ljp_cache_path,
)

model = dict(
    type='SGTFGCNRecognizer',

    teacher_backbone=dict(
        type='CTRGCNTeacher',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        clip_dim=512,           # CLIP ViT-B/32
        num_classes=60,
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        sgtf_d_h=64,
        sgtf_mlp_hidden=256,
        fusion_alpha=0.5,
        fusion_beta=0.5,
        learn_fusion_scalars=False,
        num_person=2,
    ),

    student_backbone=dict(
        type='CTRGCNStudent',
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

    # Distillation hyper-parameters (from paper / ablation)
    lambda1=0.5,
    lambda2=1.0,
    tau_TKD=2.0,
    tau_KD=4.0,
)

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']),
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
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
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file,
                     pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file,
             pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file,
              pipeline=test_pipeline, split='xsub_val'),
)

# -----------------------------------------------------------------------
# Optimizer & LR schedule
# -----------------------------------------------------------------------
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/sgtfgcn/ntu60_xsub_j_gpt55'
