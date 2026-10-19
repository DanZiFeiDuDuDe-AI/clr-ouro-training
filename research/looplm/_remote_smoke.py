"""
Remote smoke test: load 4-bit Ouro-1.4B on cheems, run a minimal forward.
Phase 0B gate 1: confirms transformers 4.54.1 + bnb 0.50.1 + clean env works.
"""
import os, sys, time, torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

from clr_ouro import load_ouro, inject_film, set_state, turn_forward_v0, PlaaCore, DS, DM

def main():
    t0 = time.time()
    print("loading Ouro 4-bit...", flush=True)
    model, tok = load_ouro()
    print("loaded in %.0fs" % (time.time() - t0), flush=True)
    # VRAM
    free, total = torch.cuda.mem_get_info()
    print("VRAM: used %.2f GB / %.2f GB" % ((total - free) / 1e9, total / 1e9), flush=True)
    # architecture sanity
    m = model.model
    print("layers=%d ut_steps=%d hidden=%d heads=%d" % (
        m.config.num_hidden_layers, m.total_ut_steps,
        m.config.hidden_size, m.config.num_attention_heads), flush=True)
    # inject FiLM
    film_layers = inject_film(model)
    plaa = PlaaCore().cuda()
    S = plaa.init_state(1, "cuda")
    set_state(film_layers, S)
    # minimal forward: single token prompt
    inp = tok("A is mother of B. Who is mother of B?", return_tensors="pt").to("cuda")
    print("input ids shape:", inp["input_ids"].shape, flush=True)
    t1 = time.time()
    with torch.no_grad():
        h, logits, h_last = turn_forward_v0(model, inp, film_layers, S)
    print("forward %.2fs, logits shape %s, argmax token %d" % (
        time.time() - t1, tuple(logits.shape), int(logits.argmax())), flush=True)
    print("argmax decoded:", repr(tok.decode([int(logits.argmax())])), flush=True)
    # check 3-tuple from inner model
    base_out, hs_list, gate_list = model.model(
        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], use_cache=False)
    print("inner 3-tuple: %d hidden_states, %d gates" % (len(hs_list), len(gate_list)), flush=True)
    print("SMOKE OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
