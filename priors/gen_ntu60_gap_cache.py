import os, sys, json, time, pickle, torch
import torch.nn.functional as F

SAVE_DIR     = "sgtfgcn_release/priors/cache"
DESC_JSON    = os.path.join(SAVE_DIR, "gpt4o_descriptions.json")
GAP_CACHE_PT = os.path.join(SAVE_DIR, "gap_cache.pt")
DATASET_PKL  = "data/nturgbd/ntu60_3danno.pkl"

API_KEY  = os.environ.get("OPENAI_API_KEY", "")  # set via environment variable, e.g. export OPENAI_API_KEY=sk-...
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL    = "gpt-4o"

NTU60_CLASSES = [
    'drink water',
    'eat meal or snack',
    'brushing teeth',
    'brushing hair',
    'drop',
    'pickup',
    'throw',
    'sitting down',
    'standing up from sitting position',
    'clapping',
    'reading',
    'writing',
    'tear up paper',
    'wear jacket',
    'take off jacket',
    'wear a shoe',
    'take off a shoe',
    'wear on glasses',
    'take off glasses',
    'put on a hat or cap',
    'take off a hat or cap',
    'cheer up',
    'hand waving',
    'kicking something',
    'reach into pocket',
    'hopping one foot jumping',
    'jump up',
    'make a phone call or answer phone',
    'playing with phone or tablet',
    'typing on a keyboard',
    'pointing to something with finger',
    'taking a selfie',
    'check time from watch',
    'rub two hands together',
    'nod head or bow',
    'shake head',
    'wipe face',
    'salute',
    'put the palms together',
    'cross hands in front',
    'sneeze or cough',
    'staggering',
    'falling',
    'touch head headache',
    'touch chest chest pain',
    'touch back backache',
    'touch neck neck ache',
    'nausea or vomiting condition',
    'use a fan with hand or paper feeling warm',
    'punching or slapping other person',
    'kicking other person',
    'pushing other person',
    'pat on back of other person',
    'point finger at the other person',
    'hugging other person',
    'giving something to other person',
    'touch other person pocket',
    'handshaking',
    'walking towards each other',
    'walking apart from each other',
]
assert len(NTU60_CLASSES) == 60


SYSTEM_PROMPT = (
    "You are an expert in human motion analysis for skeleton-based action recognition. "
    "Generate rich, structured descriptions of human actions that explicitly describe "
    "which body joints are involved, how they move, and any interaction patterns."
)


def make_user_prompt(action_name):
    parts = [
        "Describe the skeleton action: " + action_name,
        "Provide exactly three labeled parts:",
        "Body parts: describe the key joints and body segments actively involved",
        "Motion pattern: describe the movement trajectory, rhythm, and direction",
        "Interaction context: describe spatial relationships, symmetry, or person/object interaction",
        "Keep each part to 1-2 sentences. No extra text.",
    ]
    return chr(10).join(parts)


def generate_descriptions(existing):
    import openai
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    descriptions = dict(existing)
    for idx, action in enumerate(NTU60_CLASSES):
        key = str(idx)
        if key in descriptions:
            print("  [skip] %2d: %s" % (idx, action))
            continue
        print("  [gen ] %2d: %s ..." % (idx, action), end="", flush=True)
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": make_user_prompt(action)},
                    ],
                    max_tokens=300,
                    temperature=0.2,
                )
                desc = resp.choices[0].message.content.strip()
                descriptions[key] = {
                    "label_index": idx,
                    "class_name":  action,
                    "description": desc,
                    "model":       resp.model,
                }
                print(" OK (%d chars)" % len(desc))
                break
            except Exception as e:
                wait = 2 ** attempt
                print(" ERR(%s), retry..." % str(e)[:40], end="", flush=True)
                time.sleep(wait)
        else:
            print(" FAILED")
        with open(DESC_JSON, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)
        time.sleep(0.5)
    return descriptions


def build_gap_cache(descriptions):
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[CLIP] Loading ViT-B/32 on %s ..." % device)
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    embeddings = []
    for idx in range(60):
        key = str(idx)
        if key not in descriptions:
            raise ValueError("Missing description for class %d" % idx)
        full_desc = descriptions[key]["description"]
        with torch.no_grad():
            tok = clip.tokenize([full_desc], truncate=True).to(device)
            feat = model_clip.encode_text(tok)
            feat = F.normalize(feat, dim=-1)
        embeddings.append(feat.cpu())
        print("  [CLIP] %2d: %-40s shape=%s" % (idx, NTU60_CLASSES[idx][:40], str(tuple(feat.shape))))
    gap_cache = torch.cat(embeddings, dim=0)
    assert gap_cache.shape == (60, 512), "Shape mismatch: %s" % str(tuple(gap_cache.shape))
    return gap_cache


def verify_alignment(gap_cache, descriptions):
    print("\n[Verify] Checking label alignment ...")
    pkl_path = DATASET_PKL
    if not os.path.exists(pkl_path):
        pkl_path = "data/nturgbd/ntu60_hrnet.pkl"
    if not os.path.exists(pkl_path):
        print("  [WARN] No pkl found, skipping.")
        return True
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    annotations = data.get("annotations", [])
    dataset_labels = sorted(set(a["label"] for a in annotations))
    print("  Dataset labels: %d - %d, count=%d" % (min(dataset_labels), max(dataset_labels), len(dataset_labels)))
    ok = True
    for lbl in dataset_labels:
        if lbl < 0 or lbl >= 60:
            print("  [ERROR] Dataset label %d outside [0,59]" % lbl)
            ok = False
    assert gap_cache.shape == (60, 512)
    print("  Cache shape: %s  OK" % str(tuple(gap_cache.shape)))
    if torch.isnan(gap_cache).any():
        print("  [ERROR] NaN in cache")
        ok = False
    if torch.isinf(gap_cache).any():
        print("  [ERROR] Inf in cache")
        ok = False
    norms = gap_cache.norm(dim=-1)
    print("  Row norms: min=%.5f  max=%.5f" % (norms.min().item(), norms.max().item()))
    if torch.allclose(norms.float(), torch.ones(60), atol=1e-4):
        print("  All norms approx 1.0  OK")
    else:
        print("  [WARN] Some rows not unit-norm")
    print("\n  Label->Description cross-check (first 5, last 5):")
    print("  %4s  %-44s  %s" % ("idx", "class_name", "desc_start"))
    print("  " + "-" * 100)
    for i in list(range(5)) + list(range(55, 60)):
        dp = descriptions[str(i)]["description"].replace("\n", " | ")[:60]
        print("  %4d  %-44s  %s..." % (i, NTU60_CLASSES[i][:44], dp))
    bad = []
    for lbl in dataset_labels:
        cache_class = NTU60_CLASSES[lbl]
        desc_class  = descriptions[str(lbl)]["class_name"]
        if cache_class.strip().lower() != desc_class.strip().lower():
            bad.append((lbl, cache_class, desc_class))
    if bad:
        print("\n  [ERROR] Class name mismatch:")
        for lbl, cc, dc in bad:
            print("    label %d: cache=%r  desc=%r" % (lbl, cc, dc))
        ok = False
    else:
        print("\n  All class names match  OK")
    if ok:
        print("\n  [OK] Alignment verified: no mismatch detected.")
    else:
        print("\n  [FAIL] Alignment errors found.")
    return ok


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 70)
    print("Step 1: Generating NTU60 descriptions with GPT-4o")
    print("=" * 70)
    existing = {}
    if os.path.exists(DESC_JSON):
        with open(DESC_JSON, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print("  Loaded %d existing descriptions" % len(existing))
    descriptions = generate_descriptions(existing)
    with open(DESC_JSON, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    print("\n  Saved %d descriptions -> %s" % (len(descriptions), DESC_JSON))
    d = descriptions["0"]
    print("\n=== Sample class 0 ===")
    print("[0] %s\n%s" % (d["class_name"], d["description"]))
    print("\n" + "=" * 70)
    print("Step 2: Building GAP cache (CLIP ViT-B/32)")
    print("=" * 70)
    gap_cache = build_gap_cache(descriptions)
    torch.save(gap_cache, GAP_CACHE_PT)
    print("\n  Saved gap_cache %s -> %s" % (str(tuple(gap_cache.shape)), GAP_CACHE_PT))
    print("\n" + "=" * 70)
    print("Step 3: Verifying label alignment")
    print("=" * 70)
    ok = verify_alignment(gap_cache, descriptions)
    print("\n" + "=" * 70)
    print("Result: %s" % ("SUCCESS" if ok else "FAILED"))
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
