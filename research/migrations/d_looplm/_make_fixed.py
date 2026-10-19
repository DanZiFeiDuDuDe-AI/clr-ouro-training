from pathlib import Path
p = Path(r'D:\LoopLM\remote_mirror\p0h2_v2_train.py')
s = p.read_text(encoding='utf-8')

old = '''CACHE_DIR = "D:/training/teacher_cache_v2"
BASE_CKPT = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
OUT_DIR = "D:/training/checkpoints/ouro_retention_m2_h2v2"
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")
MONITOR_PATH = os.path.join(OUT_DIR, "monitor_history.json")
SEED = 42

N_STEPS = int(os.environ.get("H2_STEPS", "1500"))
LR = float(os.environ.get("H2_LR", "1e-4"))
RANK = int(os.environ.get("H2_RANK", "4"))
ALPHA = float(os.environ.get("H2_ALPHA", "8"))
LAM_BLOCK = float(os.environ.get("H2_LAM_BLOCK", "0.3"))
LAM_LOGIT = float(os.environ.get("H2_LAM_LOGIT", "1.0"))
LAM_RANK = float(os.environ.get("V2_LAM_RANK", "0.1"))
RANK_MARGIN = float(os.environ.get("V2_RANK_MARGIN", "1.0"))
RANK_MODE = os.environ.get("V2_RANK_MODE", "hinge")   # hinge | ce
MAX_EPOCHS = int(os.environ.get("V2_MAX_EPOCHS", "3"))
MONITOR_EVERY = int(os.environ.get("H2_MONITOR_EVERY", "200"))
N_EVAL = int(os.environ.get("H2_N_EVAL", "150"))
LOGIT_N = int(os.environ.get("H2_LOGIT_N", "20"))'''

new = '''CACHE_DIR = "D:/training/teacher_cache_v2"
BASE_CKPT = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
OUT_DIR = os.environ.get("H2_OUT_DIR", "D:/training/checkpoints/ouro_retention_m2_h2v2")
SEED = 42

N_STEPS = int(os.environ.get("H2_STEPS", "1500"))
LR = float(os.environ.get("H2_LR", "1e-4"))
RANK = int(os.environ.get("H2_RANK", "4"))
ALPHA = float(os.environ.get("H2_ALPHA", "8"))
LAM_BLOCK = float(os.environ.get("H2_LAM_BLOCK", "0.3"))
LAM_LOGIT = float(os.environ.get("H2_LAM_LOGIT", "1.0"))
LAM_RANK = float(os.environ.get("V2_LAM_RANK", "0.1"))
RANK_MARGIN = float(os.environ.get("V2_RANK_MARGIN", "1.0"))
RANK_MODE = os.environ.get("V2_RANK_MODE", "hinge")   # hinge | ce
MAX_EPOCHS = int(os.environ.get("V2_MAX_EPOCHS", "3"))
MONITOR_EVERY = int(os.environ.get("H2_MONITOR_EVERY", "200"))
N_EVAL = int(os.environ.get("H2_N_EVAL", "150"))
LOGIT_N = int(os.environ.get("H2_LOGIT_N", "20"))
MONITOR_SEED = int(os.environ.get("V2_MONITOR_SEED", "777"))'''
assert old in s
s = s.replace(old, new)

# Replace main seed/data block.
old2 = '''    # --- fixed seed (CRITICAL: before any generate_dataset / LoRA init) ---
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading student base...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    freeze_all(model)
    load_base(model, tok, converted)
    params = attach_lora(converted, RANK, ALPHA)
    opt = torch.optim.Adam(params, lr=LR)

    print("loading teacher (M0) for fidelity monitor...", flush=True)
    teacher_model, _ = load_ouro()
    teacher_model.eval()

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    print("cache shards: %d" % len(shards), flush=True)
    if len(shards) == 0:
        print("ERROR: empty cache dir %s — run p0h2_v2_cache.py first" % CACHE_DIR, flush=True)
        return 2

    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    # --- dataset draw, EXACT same seed stream as cache generation ---
    print("dataset draw (P0H5-aligned, held-out skipped in cache)...", flush=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _ = generate_dataset(600)                # replay parity
    _ = generate_dataset(150)[:150]          # P0H5 held-out placeholder (skip)
    corpus = generate_dataset(len(shards))[:len(shards)]
    # eval draw: FRESH seed stream -> exact P0H5 protocol (600 -> 150[:150]).
    # This is a DIFFERENT segment of the RNG stream than corpus (corpus sits
    # after the 150-placeholder), so eval never overlaps training data.
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _ = generate_dataset(600)                # replay parity
    eval_data = generate_dataset(N_EVAL)[:N_EVAL]
    print("train traj: %d | eval traj: %d" % (len(corpus), len(eval_data)), flush=True)'''

new2 = '''    # --- fixed seeds (CRITICAL: before any generate_dataset / LoRA init) ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    print("cache shards: %d" % len(shards), flush=True)
    if len(shards) == 0:
        print("ERROR: empty cache dir %s — run p0h2_v2_cache.py first" % CACHE_DIR, flush=True)
        return 2

    # --- dataset draw, EXACT same seed stream as p0h2_v2_cache.py ---
    # protocol: seed42 -> generate_dataset(600) replay -> generate_dataset(150)[:150]
    #           P0H5 held-out placeholder (skipped, NEVER cached/trained) ->
    #           corpus = generate_dataset(600) (cached in teacher_cache_v2)
    print("dataset draw (training corpus aligned with cache; P0H5 held-out skipped)...", flush=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _ = generate_dataset(600)                # replay parity
    _ = generate_dataset(150)[:150]          # P0H5 held-out placeholder (skip)
    corpus = generate_dataset(len(shards))[:len(shards)]

    # Independent monitor set (fresh, NON-42 seed) — P0H5 N=150 is final
    # held-out and must NEVER be touched during training or tuning.
    random.seed(MONITOR_SEED)
    eval_data = generate_dataset(N_EVAL)[:N_EVAL]
    print("train traj: %d | monitor traj: %d (seed=%d, independent of P0H5)" % (
        len(corpus), len(eval_data), MONITOR_SEED), flush=True)

    print("loading student base...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    freeze_all(model)
    load_base(model, tok, converted)
    # deterministic LoRA init (torch RNG may have been advanced by model loading)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    params = attach_lora(converted, RANK, ALPHA)
    opt = torch.optim.Adam(params, lr=LR)

    print("loading teacher (M0) for fidelity monitor...", flush=True)
    teacher_model, _ = load_ouro()
    teacher_model.eval()

    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]'''
assert old2 in s
s = s.replace(old2, new2)

# Replace PROGRESS_PATH/MONITOR_PATH references inside main with OUT_DIR-derived.
s = s.replace('''    os.makedirs(OUT_DIR, exist_ok=True)
''', '''    os.makedirs(OUT_DIR, exist_ok=True)
    PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")
    MONITOR_PATH = os.path.join(OUT_DIR, "monitor_history.json")
''')
# remove module-level PROGRESS_PATH/MONITOR_PATH? already removed in old block replacement (old had them, new doesn't). Good.

# Update references in main that use PROGRESS_PATH and MONITOR_PATH? They are now local vars. Fine.
# Update docstring line about eval never overlaps? optional.

out = Path(r'D:\LoopLM\remote_mirror\p0h2_v2_train_fixed.py')
out.write_text(s, encoding='utf-8')
print('wrote', out, len(s))
