"""Phase 0E forward smoke test for converted (retention) Ouro.
Verifies: shapes, no NaN, parallel/recurrent equivalence (Amendment 7),
state persist/reset API, VRAM. No training."""
import os, sys, time, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
os.environ["OURO_ROOT"] = "D:/training"
sys.path.insert(0, "D:/training")

from clr_ouro import load_ouro, MODEL_PATH
from retention_ouro import convert_ouro_to_retention, RetentionAttention

REPLACED = [3, 7, 11, 15, 19, 23]

def main():
    print("loading Ouro 4-bit...", flush=True)
    t0 = time.time()
    model, tok = load_ouro()
    print("loaded %.0fs" % (time.time()-t0), flush=True)
    free, total = torch.cuda.mem_get_info()
    print("VRAM after load: %.2f / %.2f GB" % ((total-free)/1e9, total/1e9), flush=True)

    print("converting layers %s to retention..." % REPLACED, flush=True)
    converted = convert_ouro_to_retention(model, REPLACED)
    print("converted %d layers" % len(converted), flush=True)
    # verify Q/K/V/O weights are the SAME objects as source (exact_copy trivially)
    for ret, idx in converted:
        src = ret._src_attn
        assert ret.q_proj.weight.data_ptr() == src.q_proj.weight.data_ptr(), "q_proj not shared!"
        assert ret.k_proj.weight.data_ptr() == src.k_proj.weight.data_ptr(), "k_proj not shared!"
        assert ret.v_proj.weight.data_ptr() == src.v_proj.weight.data_ptr(), "v_proj not shared!"
        assert ret.o_proj.weight.data_ptr() == src.o_proj.weight.data_ptr(), "o_proj not shared!"
    print("exact_copy check: Q/K/V/O weights shared (data_ptr identical) OK", flush=True)
    # gamma init multi-scale
    g = converted[0][0].gamma
    print("gamma[0..3]: %.4f %.4f %.4f %.4f  (h=0 should be ~0.969)" % (g[0],g[1],g[2],g[3]), flush=True)
    print("gamma[-1]: %.4f (should be ~1.0)" % g[-1], flush=True)

    inp = tok("A is mother of B. Who is mother of B?", return_tensors="pt").to("cuda")
    print("input shape:", inp["input_ids"].shape, flush=True)

    # forward through full converted model (use model.model like turn_forward_v0)
    print("forward (recurrent mode, full 4-loop)...", flush=True)
    t1 = time.time()
    with torch.no_grad():
        base_out, hs, gates = model.model(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            use_cache=False)
    dt = time.time() - t1
    h = base_out.last_hidden_state
    logits = model.lm_head(h[:, -1])
    print("forward %.2fs, h %s logits %s argmax=%d '%s'" % (
        dt, tuple(h.shape), tuple(logits.shape), int(logits.argmax()),
        repr(tok.decode([int(logits.argmax())]))), flush=True)
    assert not torch.isnan(h).any(), "NaN in hidden states!"
    assert not torch.isnan(logits).any(), "NaN in logits!"
    print("no NaN OK", flush=True)

    # Amendment 7: parallel/recurrent equivalence on one retention layer
    print("--- Amendment 7: parallel/recurrent equivalence ---", flush=True)
    ret0 = converted[0][0]
    # feed a random chunk through both paths
    torch.manual_seed(42)
    B, L = 1, 8
    x = torch.randn(B, L, 2048, device="cuda", dtype=torch.bfloat16) * 0.1
    # need position_embeddings; make dummy cos/sin
    cos = torch.ones(B, L, 128, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros(B, L, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        op, _ = ret0.forward(x, ret0.q_proj, ret0.k_proj, ret0.v_proj,
                             ret0.o_proj, position_embeddings=(cos, sin),
                             mode="parallel")
        or_, _ = ret0.forward(x, ret0.q_proj, ret0.k_proj, ret0.v_proj,
                              ret0.o_proj, position_embeddings=(cos, sin),
                              mode="recurrent")
    diff = (op - or_).abs().max().item()
    print("parallel vs recurrent max_abs_diff = %.2e (tol 1e-3 bf16)" % diff, flush=True)
    assert diff < 1e-3, "Amendment 7 FAILED: parallel != recurrent (%.2e)" % diff
    print("Amendment 7 OK", flush=True)

    # state persist/reset API — use forward() (does reshape + RoPE internally)
    print("--- state persist/reset API ---", flush=True)
    S0 = ret0.init_state(1)
    out1, S1 = ret0.forward(x, ret0.q_proj, ret0.k_proj, ret0.v_proj,
                           ret0.o_proj, position_embeddings=(cos, sin),
                           state=S0, mode="recurrent")
    assert S1[0].abs().sum() > 0, "state didn't update!"
    print("state updated: S1 norm = %.4f (S0 was 0)" % S1[0].abs().sum().item(), flush=True)
    # persist: feed second chunk with S1
    out2, S2 = ret0.forward(x, ret0.q_proj, ret0.k_proj, ret0.v_proj,
                            ret0.o_proj, position_embeddings=(cos, sin),
                            state=S1, mode="recurrent")
    # reset: feed with fresh state
    out2r, S2r = ret0.forward(x, ret0.q_proj, ret0.k_proj, ret0.v_proj,
                              ret0.o_proj, position_embeddings=(cos, sin),
                              state=ret0.init_state(1), mode="recurrent")
    diff_persist_reset = (out2 - out2r).abs().max().item()
    print("persist vs reset output diff = %.2e (should be >0 if state matters)" % diff_persist_reset, flush=True)

    free, total = torch.cuda.mem_get_info()
    print("VRAM peak: %.2f / %.2f GB" % ((total-free)/1e9, total/1e9), flush=True)
    print("PHASE 0E SMOKE OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
