"""
Build and save the GAP (CLIP) and LJP (BERT) semantic caches offline.

Usage:
    python configs/sgtfgcn/build_semantic_cache.py \
        --dataset nturgb+d \
        --save_dir data/semantic_cache/ntu60

Offline BERT (recommended when HF is blocked): place ``bert-base-uncased`` under e.g.
``data/hf_models/bert-base-uncased`` and run with::

    --bert_dir data/hf_models/bert-base-uncased --local_files_only

The resulting .pt files are loaded once at training start.
"""

import os
import re
import argparse
import pickle
import torch
import torch.nn.functional as F_n

# -------------------------------------------------------------------
# NTU RGB+D 60 action class names (60 classes, 1-indexed in literature)
# -------------------------------------------------------------------
NTU60_CLASSES = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up', 'clapping',
    'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
    'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses',
    'put on a hat/cap', 'take off a hat/cap', 'cheer up', 'hand waving',
    'kicking something', 'reach into pocket', 'hopping', 'jump up',
    'make a phone call/answer phone', 'playing with phone/tablet',
    'typing on a keyboard', 'pointing to something with finger',
    'taking a selfie', 'check time (from watch)', 'rub two hands together',
    'nod head/bow', 'shake head', 'wipe face', 'salute',
    'put the palms together', 'cross hands in front',
    'sneeze/cough', 'staggering', 'falling', 'touch head',
    'touch chest', 'touch back', 'touch neck', 'nausea or vomiting condition',
    'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person',
    'kicking other person', 'pushing other person', 'pat on back of other person',
    'point finger at the other person', 'hugging other person',
    'giving something to other person', 'touch other person pocket',
    'handshaking', 'walking towards each other', 'walking apart from each other',
]

# -------------------------------------------------------------------
# NTU RGB+D joint names (25 joints, 1-indexed in literature)
# Indices here are 0-indexed.
# -------------------------------------------------------------------
NTU_JOINT_NAMES = [
    'spine base', 'spine mid', 'neck', 'head',
    'left shoulder', 'left elbow', 'left wrist', 'left hand',
    'right shoulder', 'right elbow', 'right wrist', 'right hand',
    'left hip', 'left knee', 'left ankle', 'left foot',
    'right hip', 'right knee', 'right ankle', 'right foot',
    'spine', 'left hand tip', 'left thumb', 'right hand tip', 'right thumb',
]

# -------------------------------------------------------------------
# COCO 17-joint names (for HRNet-based keypoints)
# -------------------------------------------------------------------
COCO_JOINT_NAMES = [
    'nose', 'left eye', 'right eye', 'left ear', 'right ear',
    'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
    'left wrist', 'right wrist', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle',
]


# -------------------------------------------------------------------
# Diving-48 class names (label index 0-47)
# Based on FINA diving codes; Li et al. ECCV 2018 "RESOUND" dataset.
# Note: label 30 has no samples in diving48_hrnet.pkl but must be kept
# to preserve index alignment.
# -------------------------------------------------------------------
DIVING48_CLASSES = [
    'forward dive tuck',                           # 0
    'forward dive pike',                           # 1
    'forward dive straight',                       # 2
    'back dive tuck',                              # 3
    'back dive pike',                              # 4
    'back dive straight',                          # 5
    'reverse dive tuck',                           # 6
    'reverse dive pike',                           # 7
    'reverse dive straight',                       # 8
    'inward dive tuck',                            # 9
    'inward dive pike',                            # 10
    'inward dive straight',                        # 11
    'forward 1.5 somersaults tuck',                # 12
    'forward 1.5 somersaults pike',                # 13
    'forward 2 somersaults tuck',                  # 14
    'forward 2 somersaults pike',                  # 15
    'forward 2.5 somersaults tuck',                # 16
    'forward 2.5 somersaults pike',                # 17
    'forward 3 somersaults tuck',                  # 18
    'forward 3.5 somersaults tuck',                # 19
    'back 1.5 somersaults tuck',                   # 20
    'back 1.5 somersaults pike',                   # 21
    'back 2 somersaults tuck',                     # 22
    'back 2 somersaults pike',                     # 23
    'back 2.5 somersaults tuck',                   # 24
    'back 2.5 somersaults pike',                   # 25
    'back 3 somersaults tuck',                     # 26
    'back 3.5 somersaults tuck',                   # 27
    'reverse 1.5 somersaults tuck',                # 28
    'reverse 1.5 somersaults pike',                # 29
    'reverse 2 somersaults tuck',                  # 30  (no samples in pkl)
    'reverse 2.5 somersaults tuck',                # 31
    'reverse 2.5 somersaults pike',                # 32
    'inward 1.5 somersaults tuck',                 # 33
    'inward 1.5 somersaults pike',                 # 34
    'inward 2 somersaults tuck',                   # 35
    'inward 2 somersaults pike',                   # 36
    'inward 2.5 somersaults tuck',                 # 37
    'forward 1 somersault with 1 twist tuck',      # 38
    'forward 1.5 somersaults with 2 twists pike',  # 39
    'forward 2 somersaults with 2 twists pike',    # 40
    'forward 2.5 somersaults with 2 twists pike',  # 41
    'back 1.5 somersaults with 2 twists pike',     # 42
    'back 2 somersaults with 2 twists pike',       # 43
    'back 2.5 somersaults with 2 twists pike',     # 44
    'reverse 1.5 somersaults with 2 twists pike',  # 45
    'reverse 2 somersaults with 2 twists pike',    # 46
    'inward 1.5 somersaults with 2 twists pike',   # 47
]

DATASET_CONFIGS = {
    'nturgb+d': {'classes': NTU60_CLASSES, 'joints': NTU_JOINT_NAMES},
    'diving48':  {'classes': DIVING48_CLASSES,  'joints': COCO_JOINT_NAMES},
}


def _camel_case_to_words(name: str) -> str:
    """ApplyEyeMakeup -> Apply Eye Makeup (for text encoders)."""
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return spaced.replace('_', ' ').strip()


def ucf101_classes_from_ann_pkl(ann_path: str):
    """UCF101 class names in label order 0..100, aligned with pyskl PoseDataset.

    Names are parsed from ``frame_dir`` (``v_<ClassName>_g##_c##``). Human-readable
    spacing is added via CamelCase splitting for CLIP/BERT prompts.
    """
    with open(ann_path, 'rb') as f:
        data = pickle.load(f)
    ann = data['annotations']
    pat = re.compile(r'^v_(.+)_g\d+_c\d+$')

    one_per_label = {}
    for item in ann:
        lbl = item['label']
        if lbl in one_per_label:
            continue
        fd = item['frame_dir']
        base = fd.rstrip('/').split('/')[-1]
        m = pat.match(base)
        if not m:
            raise ValueError(
                f'Cannot parse UCF101 class from frame_dir={fd!r} '
                f'(expected v_<Class>_g##_c##)')
        raw_cls = m.group(1)
        one_per_label[lbl] = _camel_case_to_words(raw_cls)

    missing = [i for i in range(101) if i not in one_per_label]
    if missing:
        raise ValueError(f'Missing labels in ann file: {missing[:10]}...')

    classes = [one_per_label[i] for i in range(101)]
    return classes


def verify_ucf101_classes(ann_path: str, classes: list):
    """Sanity checks against the pickle (single source of truth)."""
    assert len(classes) == 101
    with open(ann_path, 'rb') as f:
        data = pickle.load(f)
    ann = data['annotations']
    labels_in_ann = {x['label'] for x in ann}
    assert labels_in_ann == set(range(101)), 'labels must be 0..100'

    pat = re.compile(r'^v_(.+)_g\d+_c\d+$')
    for lbl in (0, 50, 100):
        fd = next(x['frame_dir'] for x in ann if x['label'] == lbl)
        base = fd.rstrip('/').split('/')[-1]
        raw = pat.match(base).group(1)
        expected_prompt = _camel_case_to_words(raw)
        assert classes[lbl] == expected_prompt, (
            f'label {lbl}: pickle-derived {expected_prompt!r} != cache list '
            f'{classes[lbl]!r}')
    print('  Verification: UCF101 labels 0..100 and spot-checks OK.')


def build_gap_cache(classes, clip_model='ViT-B/32', device='cpu', class_prompts=None):
    """Encode 3-perspective action descriptions with CLIP.

    Args:
        classes: list of class name strings (length K)
        clip_model: CLIP variant (default ViT-B/32)
        device: torch device string
        class_prompts: optional dict mapping class_name -> dict with keys
            'body_part', 'motion_trajectory', 'interaction_context'.
            When provided (e.g. from GPT-generated JSON), these replace the
            generic templates. Classes missing from the dict fall back to templates.

    Returns:
        gap_cache: (K, 512) float32 tensor
    """
    try:
        import clip
    except ImportError:
        raise ImportError("Install openai-clip: pip install git+https://github.com/openai/CLIP.git")

    model, preprocess = clip.load(clip_model, device=device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for cls_name in classes:
            if class_prompts and cls_name in class_prompts:
                # Use LLM-generated per-class descriptions
                p = class_prompts[cls_name]
                sentences = [
                    p['body_part'],
                    p['motion_trajectory'],
                    p['interaction_context'],
                ]
            else:
                # Generic template fallback
                sentences = [
                    f"A person is performing {cls_name}. "
                    f"The active body parts are the hands and arms.",
                    f"The motion trajectory of {cls_name} involves coordinated movement "
                    f"of upper and lower limbs.",
                    f"During {cls_name}, the person interacts with objects or other people.",
                ]
            text = clip.tokenize([' '.join(sentences)]).to(device)
            emb = model.encode_text(text).float().squeeze(0)   # (512,)
            embeddings.append(emb)

    gap_cache = torch.stack(embeddings)  # (K, 512)
    return gap_cache


def _embeddings_to_ljp_adj(embeddings: torch.Tensor) -> torch.Tensor:
    """Match ``LJPCache.set_cache``: L2-normalize rows then Gram matrix per class."""
    h = F_n.normalize(embeddings.float(), p=2, dim=-1)
    return torch.bmm(h, h.transpose(1, 2))


def build_ljp_cache(classes, joint_names, device='cpu',
                    bert_model_name='bert-base-uncased',
                    bert_dir=None,
                    local_files_only=False,
                    ljp_backend='bert'):
    """Encode structured joint-function prompts with BERT.

    Returns:
        ljp_cache: (K, V, V) float32 tensor  (cosine similarity Gram matrices)
    """
    K = len(classes)
    V = len(joint_names)
    def _build_ljp_clip():
        import clip
        clip_model, _ = clip.load('ViT-B/32', device=device)
        clip_model.eval()
        emb = torch.zeros(K, V, 512)
        with torch.no_grad():
            for k, cls_name in enumerate(classes):
                for i, joint_name in enumerate(joint_names):
                    prompt = f"{joint_name} function in {cls_name}"
                    text = clip.tokenize([prompt]).to(device)
                    emb[k, i] = clip_model.encode_text(text).float().squeeze(0).cpu()
        print("  LJP backend: CLIP (ViT-B/32)")
        return emb

    # Primary path: BERT embeddings (paper-consistent).
    # clip path: no HuggingFace (offline-friendly).
    # bert path + failure: fall back to CLIP.
    if ljp_backend == 'clip':
        emb_mat = _build_ljp_clip()
        return _embeddings_to_ljp_adj(emb_mat)

    emb_mat = None
    try:
        from transformers import BertTokenizer, BertModel
        model_source = bert_dir if bert_dir else bert_model_name
        tokenizer = BertTokenizer.from_pretrained(
            model_source, local_files_only=local_files_only)
        bert = BertModel.from_pretrained(
            model_source, local_files_only=local_files_only).to(device)
        bert.eval()

        emb_mat = torch.zeros(K, V, 768)
        with torch.no_grad():
            for k, cls_name in enumerate(classes):
                for i, joint_name in enumerate(joint_names):
                    prompt = f"{joint_name} function in {cls_name}"
                    enc = tokenizer(prompt, return_tensors='pt',
                                    truncation=True, max_length=64).to(device)
                    out = bert(**enc)
                    emb_mat[k, i] = out.pooler_output.squeeze(0).cpu()
        print("  LJP backend: BERT (bert-base-uncased)")
    except Exception as e:
        print(f"  Warning: BERT backend unavailable ({e})")
        print("  Falling back to CLIP text embeddings for LJP cache generation.")
        emb_mat = _build_ljp_clip()

    return _embeddings_to_ljp_adj(emb_mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nturgb+d',
                        choices=list(DATASET_CONFIGS.keys()) + ['ucf101'])
    parser.add_argument(
        '--ann_file',
        default=None,
        help='UCF101: path to ucf101_hrnet.pkl (required if dataset=ucf101)')
    parser.add_argument(
        '--prompts_file',
        default=None,
        help='Path to JSON file with GPT-generated per-class prompts '             '(e.g. data/semantic_cache/diving48/diving48_prompts.json). '             'When provided, these replace generic templates in the GAP cache.')
    parser.add_argument('--save_dir', default='data/semantic_cache/ntu60')
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--bert_model_name', default='bert-base-uncased',
                        help='HF model id for BERT when not using --bert_dir')
    parser.add_argument('--bert_dir', default=None,
                        help='Local directory containing BERT tokenizer/model files')
    parser.add_argument('--local_files_only', action='store_true',
                        help='Force transformers to load from local cache/files only')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        if not args.ann_file:
            raise ValueError('dataset=ucf101 requires --ann_file path/to/ucf101_hrnet.pkl')
        classes = ucf101_classes_from_ann_pkl(args.ann_file)
        joints = COCO_JOINT_NAMES
        verify_ucf101_classes(args.ann_file, classes)
    else:
        cfg = DATASET_CONFIGS[args.dataset]
        classes = cfg['classes']
        joints = cfg['joints']

    # Load optional per-class GPT prompts
    class_prompts = None
    if args.prompts_file:
        import json as _json
        with open(args.prompts_file, encoding='utf-8') as _f:
            class_prompts = _json.load(_f)
        covered = sum(1 for c in classes if c in class_prompts)
        print(f"  Loaded {len(class_prompts)} GPT prompts from {args.prompts_file} "
              f"({covered}/{len(classes)} classes covered)")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Building GAP cache for {len(classes)} classes ...")
    gap = build_gap_cache(classes, args.clip_model, args.device, class_prompts=class_prompts)
    gap_path = os.path.join(args.save_dir, 'gap_cache.pt')
    torch.save(gap, gap_path)
    print(f"  Saved GAP cache ({gap.shape}) -> {gap_path}")

    print(f"Building LJP cache for {len(classes)} classes x {len(joints)} joints ...")
    ljp = build_ljp_cache(
        classes, joints, args.device,
        bert_model_name=args.bert_model_name,
        bert_dir=args.bert_dir,
        local_files_only=args.local_files_only)
    ljp_path = os.path.join(args.save_dir, 'ljp_cache.pt')
    torch.save(ljp, ljp_path)
    print(f"  Saved LJP cache ({ljp.shape}) -> {ljp_path}")


if __name__ == '__main__':
    main()
