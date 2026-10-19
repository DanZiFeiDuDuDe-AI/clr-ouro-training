# -*- coding: utf-8 -*-
"""
M1 day-1 gate: can Ouro-1.4B load in 4-bit on the GTX 1650 under
transformers 5.13, and does the UT-loop forward behave as documented?

Checks:
  1. 4-bit NF4 load via trust_remote_code (modeling_ouro targets tf 4.55 APIs).
  2. VRAM after load.
  3. Inner OuroModel returns (BaseModelOutputWithPast, hidden_states_list, gate_list)
     with len == total_ut_steps == 4  (the CLR per-loop hook point).
  4. Greedy generation of a few tokens (end-to-end sanity).
"""
import torch, os, sys, time

MODEL = "D:/LoopLM_models/Ouro-1.4B"

def vram(tag):
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    print(f"[VRAM] {tag}: allocated={a:.2f}GB reserved={r:.2f}GB", flush=True)

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    print("transformers", transformers.__version__, flush=True)

    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=bnb, device_map={"": "cuda:0"}, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    vram("after load")
    print("total_ut_steps:", model.config.total_ut_steps, flush=True)
    print("early_exit_gate:", model.model.early_exit_gate, flush=True)

    # --- 3) inner-model tuple output (the CLR hook point) ---
    inp = tok("Fact: Anna is the mother of Ben.\nQuestion: What is Anna to Ben?\nAnswer:",
              return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.model(input_ids=inp["input_ids"],
                          attention_mask=inp["attention_mask"], use_cache=False)
    assert isinstance(out, tuple) and len(out) == 3, f"unexpected output type: {type(out)}"
    base_out, hs_list, gate_list = out
    print(f"inner model returns tuple: hidden_states_list={len(hs_list)} entries, "
          f"gate_list={len(gate_list)} entries", flush=True)
    print("last_hidden_state shape:", tuple(base_out.last_hidden_state.shape), flush=True)
    lam = [float(torch.sigmoid(g[0, -1, 0])) for g in gate_list]
    print("exit-gate sigmoids at last token per UT step:", [round(x, 3) for x in lam], flush=True)

    # next-token argmax from final-step hidden (the v0 CLR eval path)
    with torch.no_grad():
        logits = model.lm_head(base_out.last_hidden_state[:, -1])
    print("next-token argmax:", repr(tok.decode(logits.argmax(-1))), flush=True)
    vram("after forward")

    # --- 4) generation sanity ---
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=12, do_sample=False)
    print("generation:", repr(tok.decode(gen[0][inp['input_ids'].shape[1]:],
          skip_special_tokens=True)), flush=True)
    vram("after generate")
    print("SMOKE TEST PASSED", flush=True)

if __name__ == "__main__":
    main()
