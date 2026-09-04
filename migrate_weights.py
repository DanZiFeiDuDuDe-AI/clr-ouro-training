# -*- coding: utf-8 -*-
"""
Phase 0F — weight migration: save M1 checkpoint + manifest + verify.

Loads Ouro-1.4B (4-bit frozen), converts layers {3,7,11,15,19,23} to retention
(retention_ouro.convert_ouro_to_retention), then:
  1. exact-copy verification: for every exact_copy tensor, compare source
     (pre-conversion) vs target (post-conversion) values -> max_abs_diff==0.
  2. UNCHANGED_PARAMETER_CHECK: all 18 unchanged layers' tensors match source.
  3. Save M1 checkpoint (retention gamma + manifest pointer; frozen backbone
     stays referenced by path, not re-saved).
  4. Write weight_migration_manifest.json (P0_WEIGHT_MIGRATION.md §4 schema).

Amendment 3: exact_copy means tensor copied exactly, NOT attention function
preserved. Manifest notes reflect this.
"""
import os, sys, json, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention

REPLACED = [3, 7, 11, 15, 19, 23]
SOURCE_HASH = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
MANIFEST_PATH = "D:/training/weight_migration_manifest.json"
M1_CKPT_PATH = "D:/training/checkpoints/ouro_retention_m1/best_model.pt"

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()[:16]


def _resolve_target_tensor(model, name):
    """Resolve a snapshot name to its current tensor on the (converted) model.

    Name forms:
      layers.{i}.self_attn.{q,k,v,o}_proj.weight
      layers.{i}.{input_layernorm|input_layernorm_2|post_attention_layernorm|post_attention_layernorm_2}.weight
      layers.{i}.mlp.{gate_proj|up_proj|down_proj}.weight
      model.norm.weight / model.early_exit_gate.weight / model.early_exit_gate.bias
      embed_tokens.weight / lm_head.weight
    """
    parts = name.split(".")
    if parts[0] == "layers":
        # layers.{i}.<rest>
        layer_i = int(parts[1])
        layer = model.model.layers[layer_i]
        rest = parts[2:]
        if rest[0] == "self_attn":
            attn = layer.self_attn
            src = getattr(attn, "src_attn", attn)  # adapter -> original OuroAttention
            proj = getattr(src, rest[1])           # q_proj / k_proj / v_proj / o_proj
            return proj.weight
        elif rest[0] == "mlp":
            proj = getattr(layer.mlp, rest[1])
            return proj.weight
        else:
            # a norm: input_layernorm / input_layernorm_2 / post_attention_layernorm(_2)
            norm = getattr(layer, rest[0])
            return norm.weight
    elif parts[0] == "model":
        # model.norm.weight / model.early_exit_gate.weight|bias
        obj = getattr(model, "model")
        sub = getattr(obj, parts[1])              # norm / early_exit_gate
        if parts[2] == "weight":
            return sub.weight
        else:                                      # bias
            return sub.bias
    elif parts[0] == "embed_tokens":
        return model.model.embed_tokens.weight
    elif parts[0] == "lm_head":
        return model.lm_head.weight
    raise KeyError("unresolved name: " + name)


def main():
    print("=== Phase 0F: weight migration ===", flush=True)
    # Snapshot source state BEFORE conversion: capture q/k/v/o/norm/mlp tensors
    # for every layer. Since conversion keeps the SAME nn.Linear modules (just
    # routes forward through retention), the tensors are literally shared.
    # We snapshot their values + checksums for the manifest.
    print("loading Ouro 4-bit...", flush=True)
    t0 = time.time()
    model, tok = load_ouro()
    print("loaded %.0fs" % (time.time()-t0), flush=True)

    layers = model.model.layers
    num_layers = len(layers)  # 24

    # Pre-conversion snapshot: capture all exact_copy tensors' values+checksums.
    # These are the SOURCE values. After conversion they must be identical.
    print("snapshotting source tensors...", flush=True)
    src_snap = {}  # name -> (value_clone, checksum, shape)
    for i in range(num_layers):
        la = layers[i].self_attn
        for pn in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = getattr(la, pn).weight
            src_snap[f"layers.{i}.self_attn.{pn}.weight"] = (w.detach().clone(), tensor_sha(w), tuple(w.shape))
        ln = layers[i]
        for pn in ["input_layernorm", "input_layernorm_2",
                   "post_attention_layernorm", "post_attention_layernorm_2"]:
            w = getattr(ln, pn).weight
            src_snap[f"layers.{i}.{pn}.weight"] = (w.detach().clone(), tensor_sha(w), tuple(w.shape))
        for pn in ["gate_proj", "up_proj", "down_proj"]:
            w = getattr(ln.mlp, pn).weight
            src_snap[f"layers.{i}.mlp.{pn}.weight"] = (w.detach().clone(), tensor_sha(w), tuple(w.shape))
    # model-level
    src_snap["model.norm.weight"] = (model.model.norm.weight.detach().clone(), tensor_sha(model.model.norm.weight), tuple(model.model.norm.weight.shape))
    src_snap["model.early_exit_gate.weight"] = (model.model.early_exit_gate.weight.detach().clone(), tensor_sha(model.model.early_exit_gate.weight), tuple(model.model.early_exit_gate.weight.shape))
    src_snap["model.early_exit_gate.bias"] = (model.model.early_exit_gate.bias.detach().clone(), tensor_sha(model.model.early_exit_gate.bias), tuple(model.model.early_exit_gate.bias.shape))
    src_snap["embed_tokens.weight"] = (model.model.embed_tokens.weight.detach().clone(), tensor_sha(model.model.embed_tokens.weight), tuple(model.model.embed_tokens.weight.shape))
    src_snap["lm_head.weight"] = (model.lm_head.weight.detach().clone(), tensor_sha(model.lm_head.weight), tuple(model.lm_head.weight.shape))
    print("  snapshot %d tensors" % len(src_snap), flush=True)

    # Convert
    print("converting %d layers to retention..." % len(REPLACED), flush=True)
    converted = convert_ouro_to_retention(model, REPLACED)
    ret_by_layer = {idx: ret for ret, idx in converted}

    # Post-conversion verification: every src_snap tensor must be unchanged.
    print("=== exact-copy verification ===", flush=True)
    mismatch = 0
    max_diff_global = 0.0
    manifest_layers = []
    for name, (src_val, src_cs, src_shape) in src_snap.items():
        # fetch target tensor by name — handle the path forms we snapshotted.
        # 'layers' prefix -> model.model.layers (ModuleList, integer index).
        # Replaced layers' self_attn is an adapter; projections live on
        # its .src_attn (the original attention module).
        tgt = _resolve_target_tensor(model, name)
        diff = (src_val.float() - tgt.detach().float()).abs().max().item()
        max_diff_global = max(max_diff_global, diff)
        if diff > 0:
            mismatch += 1
            print("  MISMATCH %s diff=%.2e" % (name, diff), flush=True)
        # build manifest entry
        parts = name.split(".")
        layer_idx = int(parts[1]) if parts[0] == "layers" else -1
        is_replaced = layer_idx in REPLACED
        # q/k/v/o in replaced layers: exact_copy but mixing changed
        if is_replaced and "self_attn" in name and any(p in name for p in ["q_proj","k_proj","v_proj","o_proj"]):
            note = "exact tensor copy; projection role preserved, mixing changed softmax->retention (Amendment 3)"
        elif is_replaced and "self_attn" in name:
            note = "exact_copy"
        else:
            note = "exact_copy (unchanged layer)"
        manifest_layers.append({
            "source": name, "target": name, "mode": "exact_copy",
            "source_shape": list(src_shape), "target_shape": list(tgt.shape),
            "rule": "direct copy" if diff == 0 else "MISMATCH",
            "checksum_before": src_cs, "checksum_after": tensor_sha(tgt),
            "notes": note})

    print("  %d tensors checked, %d mismatch, max_abs_diff=%.2e" % (
        len(src_snap), mismatch, max_diff_global), flush=True)
    assert mismatch == 0, "exact-copy verification FAILED: %d mismatches" % mismatch
    assert max_diff_global == 0.0, "max_abs_diff != 0: %.2e" % max_diff_global
    print("  exact-copy OK (max_abs_diff == 0)", flush=True)

    # new parameters: gamma_logit per replaced layer
    print("=== new parameters (gamma) ===", flush=True)
    for ret, idx in converted:
        gname = f"layers.{idx}.self_attn.retention.gamma_logit"
        manifest_layers.append({
            "source": None, "target": gname, "mode": "new_parameter",
            "source_shape": None, "target_shape": list(ret.gamma_logit.shape),
            "rule": "multi-scale init gamma_h=sigmoid(theta_h), theta_h=logit(1-2^(-5-h))",
            "checksum_before": None, "checksum_after": tensor_sha(ret.gamma_logit),
            "notes": "only new trainable param; per-head decay (Amendment 1)"})

    # Save M1 checkpoint (only retention gamma + converter metadata; backbone
    # frozen 4-bit weights are NOT re-saved — they are loaded from source path).
    print("saving M1 checkpoint...", flush=True)
    os.makedirs(os.path.dirname(M1_CKPT_PATH), exist_ok=True)
    m1_state = {
        "variant": "ouro_retention_m1",
        "replaced_layers": REPLACED,
        "retention_gamma": {str(idx): ret.gamma_logit.detach().cpu().clone()
                            for ret, idx in converted},
        "source_checkpoint_hash": SOURCE_HASH,
        "source_model_path": "D:/training/Ouro-1.4B",
        "note": "M1: architecture converted + migrated init, NO distillation. "
                "Backbone weights are exact_copy from source (not re-saved); "
                "only gamma_logit is new.",
    }
    torch.save(m1_state, M1_CKPT_PATH)
    print("  saved %s (%d bytes)" % (M1_CKPT_PATH, os.path.getsize(M1_CKPT_PATH)), flush=True)

    # Summary
    n_exact = sum(1 for e in manifest_layers if e["mode"] == "exact_copy")
    n_new = sum(1 for e in manifest_layers if e["mode"] == "new_parameter")
    n_exact_params = sum(src_snap[k][0].numel() for k in src_snap)
    n_new_params = n_new * 16  # 16 heads per replaced layer
    total_params = n_exact_params + n_new_params
    manifest = {
        "source_model": "ByteDance/Ouro-1.4B",
        "target_model": "Ouro-1.4B-retention-6layer-hybrid",
        "source_checkpoint_hash": SOURCE_HASH,
        "target_initial_checkpoint_hash": "",  # filled below
        "config": {
            "replaced_layers": REPLACED, "replacement_ratio": "6/24 = 25%",
            "mixing": "linear_retention_with_rope",
            "state_reset": "per_ouro_loop_boundary (Amendment 5)",
            "position_semantics": "episode_global (Amendment 4)",
            "gamma_init": "multi_scale_1-2^(-5-h)",
        },
        "layers": manifest_layers,
        "summary": {
            "pct_exact_copy": "%.4f%%" % (100.0 * n_exact_params / total_params),
            "pct_structured_init": "0%",
            "pct_new_parameter": "%.4f%% (%d params, gamma only)" % (100.0 * n_new_params / total_params, n_new_params),
            "exact_copy_tensor_count": n_exact,
            "exact_copy_param_count": n_exact_params,
            "new_param_count": n_new_params,
            "mismatch_count": mismatch,
            "max_abs_diff_exact_copy": max_diff_global,
        },
    }
    # real target ckpt hash
    with open(M1_CKPT_PATH, "rb") as f:
        manifest["target_initial_checkpoint_hash"] = hashlib.sha256(f.read()).hexdigest()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote manifest %s" % MANIFEST_PATH, flush=True)
    print("  exact_copy tensors: %d (%.0f params)" % (n_exact, n_exact_params), flush=True)
    print("  new_parameter tensors: %d (%d params, gamma)" % (n_new, n_new*16), flush=True)
    print("  structured_init: 0", flush=True)
    print("  mismatch: %d, max_abs_diff: %.2e" % (mismatch, max_diff_global), flush=True)
    print("\nPHASE 0F OK — migration manifest written, exact-copy verified", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
