# SGTF-GCN: Semantics-Guided Topology Fusion for Skeleton-Based Action Recognition

---

## Overview

**SGTF-GCN** introduces two complementary semantic priors into graph convolutional networks for skeleton-based action recognition, coupled with a **Topology Knowledge Distillation (TKD)** training strategy that transfers semantic topology awareness from a text-guided teacher to a lightweight, inference-efficient student.

### Key Components

| Component | Description |
|---|---|
| **GAP** (Global Action-context Prior) | CLIP-encoded action descriptions modulate inter-joint attention to build a dynamic adjacency A_gap ∈ R^{N×V×V} |
| **LJP** (Local Joint-relation Prior) | Per-class BERT-encoded joint-function templates, precomputed offline, yielding A_ljp ∈ R^{V×V} per class |
| **Topology Fusion** | A_sem = D^{-1/2} (A_phy + α·A_gap + β·A_ljp) D^{-1/2} |
| **TKD** | KL-divergence alignment of softmax(A_sem/τ) and softmax(A_adp/τ) at each GCN block |

### Training / Inference Pipeline

```
Training
  ├─ Teacher (CTRGCNTeacher + SGTF)  →  L_task_teacher
  └─ Student (CTRGCN)                →  L_task_student + λ₁·L_TKD + λ₂·L_KD

Inference
  └─ Student only  (no text encoder required)
```

---

## Repository Structure

```
SGTF-GCN/
 README.md

 models/                              # Core model code
   ├── sgtf_module.py                   # SGTF module: GAPModule + LJPModule + topology fusion
   ├── sgtf_module_v2.py                # Decoupled SGTF module for the tri-modal variant
   ├── sgtfgcn_recognizer.py            # End-to-end recognizer (teacher + student + TKD loss)
   ├── trisgtfgcn_recognizer.py         # Tri-modal (J/B/JM) recognizer variant
   ├── ctrgcn_teacher.py                # Teacher backbone (CTR-GCN + SGTF hooks)
   ├── ctrgcn_teacher_tri.py            # Tri-modal teacher backbone
   ├── ctrgcn_student.py                # Student backbone
   ├── ctrgcn_student_tri.py            # Tri-modal student backbone
   ├── unit_ctrgcn_teacher_tri.py       # Tri-modal teacher GCN unit
   └── ctrgcn_kd.py                     # KD utility wrapper

 priors/                              # Semantic prior generation & loading
   ├── build_semantic_cache.py          # Main script: GPT API → LJP & GAP caches
   ├── gen_ntu60_gap_cache.py           # GAP cache generation for NTU60 (CLIP encoding)
   ├── semantic_cache.py                # Utility: load gap_cache.pt / ljp_cache.pt into model
   └── cache/
       ├── gpt4o_descriptions.json      # Pre-generated GPT-4o descriptions (NTU60, 60 classes)
       ├── gap_cache.pt                 # Pre-built GAP embeddings  [60, 512]    (float16)
       └── ljp_cache.pt                 # Pre-built LJP adjacency   [60, 25, 25] (float32)

 configs/                             # Example training configs (PySKL format)
    ├── sgtfgcn_ntu60_xsub_j.py         # NTU60 X-Sub, joint stream
    ├── sgtfgcn_ntu60_xsub_j_gpt4o.py   # NTU60 X-Sub, joint + GPT-4o descriptions
    ├── sgtfgcn_ucf101_hrnet_j.py        # UCF101, HRNet joint stream
    └── trisgtfgcn_ntu60_xsub_j.py       # NTU60 X-Sub, tri-modal fusion
```

---

## Semantic Prior Generation

The two priors are precomputed **offline** before training and stored as `.pt` tensors.  
We provide pre-built caches for NTU60 (60 classes) in `priors/cache/` — you can skip Steps 1–3 and use them directly.

### Step 1 — Generate GPT Action Descriptions

From the PySKL project root, run `gen_ntu60_gap_cache.py` to query GPT-4o and save structured descriptions to `priors/cache/gpt4o_descriptions.json`. Each entry contains:

```json
"0": {
  "label_index": 0,
  "class_name": "drink water",
  "description": "Body parts: ...\n\nMotion pattern: ...\n\nInteraction context: ...",
  "model": "gpt-4o"
}
```

```bash
python priors/gen_ntu60_gap_cache.py
```

### Step 2 — Build the LJP Cache (BERT embeddings → 25×25 adjacency)

```bash
python priors/build_semantic_cache.py \
    --dataset nturgb+d \
    --save_dir priors/cache \
    --bert_dir data/hf_models/bert-base-uncased \
    --local_files_only \
    --skip_gap
```

Output: `ljp_cache.pt` — shape `[K, 25, 25]`.

### Step 3 — Build the GAP Cache (CLIP text embeddings)

`gen_ntu60_gap_cache.py` also encodes the GPT-4o descriptions with CLIP ViT-B/32 and saves `gap_cache.pt` to `priors/cache/`.

Output: `gap_cache.pt` — shape `[K, 512]`.

### Pre-built Cache Summary (NTU60)

| File | Shape | Content |
|---|---|---|
| `cache/gap_cache.pt` | `[60, 512]` | CLIP text embeddings (float16) — **pre-built, ready to use** |
| `cache/ljp_cache.pt` | `[60, 25, 25]` | BERT-derived joint-pair adjacency (float32) — **pre-built, ready to use** |
| `cache/gpt4o_descriptions.json` | — | Structured GPT-4o descriptions for all 60 classes |

---

## Training

Copy this repository into your PySKL project (e.g. as `sgtfgcn_release/`), then specify cache paths in your config file:

```python
# inside your .py config
semantic_cache = dict(
    gap_cache_path     = 'sgtfgcn_release/priors/cache/gap_cache.pt',
    ljp_adj_cache_path = 'sgtfgcn_release/priors/cache/ljp_cache.pt',
)
```

### Paper Training Setting

The released NTU60 configs follow the paper setting: 100 epochs, 5-epoch linear warm-up, step learning-rate decay at epochs 35 and 55, SGD with Nesterov momentum 0.9, weight decay 0.0004, 64-frame input sequences, and trainable topology-fusion scalars `alpha` and `beta` initialized to 0.5.

Launch training with the PySKL runner:

```bash
# Single GPU
python tools/train.py sgtfgcn_release/configs/sgtfgcn_ntu60_xsub_j_gpt4o.py \
    --work-dir work_dirs/sgtfgcn_ntu60_xsub

# Distributed (8 GPUs)
bash tools/dist_train.sh sgtfgcn_release/configs/sgtfgcn_ntu60_xsub_j_gpt4o.py 8
```

### Loss Configuration

| Loss | Default weight | Temperature |
|---|---|---|
| Task loss — teacher | — | — |
| Task loss — student | — | — |
| Topology KD (TKD) | `lambda1 = 0.5` | `tau_TKD = 2.0` |
| Logits KD (Hinton) | `lambda2 = 1.0` | `tau_KD = 4.0` |

---

## Inference

Only the **student** network runs at test time — no CLIP or BERT required:

```bash
python tools/test.py sgtfgcn_release/configs/sgtfgcn_ntu60_xsub_j_gpt4o.py \
    work_dirs/sgtfgcn_ntu60_xsub/best.pth
```

---

## Dependencies

Built on top of [PySKL](https://github.com/kennymckormick/pyskl):

```bash
pip install -r requirements.txt
pip install -e.
```

Additional dependencies for prior generation:

```bash
pip install openai clip transformers
```

---



> **[Paper link and full citation will be updated upon acceptance.]**

---

## License

This project is released under the Apache 2.0 License.

---

## Coming Soon

Full training code, evaluation scripts, and pre-trained checkpoints will be released upon paper acceptance.
