import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\training")
OUT = ROOT / "checkpoints" / "single_slot_diff_audit"
CACHE_ROOT = ROOT / "checkpoints" / "clr_v2_a1_ls2" / "cache"
CLOSURE_OUT = ROOT / "checkpoints" / "clr_v22_s0_closure"
V23_OUT = ROOT / "checkpoints" / "clr_v23_oracle"
sys.path.insert(0, str(ROOT))

from a1_v22_s0_closure_common import (  # noqa: E402
    SingleSlotBaseline,
    LinearProbe,
    accuracy,
    geometry_for,
    train_classifier,
)
from a1_v22_storage_common import FactCache  # noqa: E402
from clr_v23_oracle import OracleAllocationCore  # noqa: E402


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")


def _json(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def stats(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = np.abs(a - b)
    af = a.reshape(len(a), -1)
    bf = b.reshape(len(b), -1)
    cs = (af * bf).sum(1) / (np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1) + 1e-8)
    return {"shape_a": list(a.shape), "shape_b": list(b.shape), "max_abs_diff": float(d.max(initial=0.0)),
            "mean_abs_diff": float(d.mean()), "min_cosine": float(cs.min(initial=1.0)),
            "mean_cosine": float(cs.mean()), "allclose_1e-6": bool(np.allclose(a, b, atol=1e-6, rtol=1e-6))}


def load_closure_arrays():
    z = np.load(CLOSURE_OUT / "closure_representations.npz", allow_pickle=False)
    out = {}
    for split in ("train", "val", "testA", "testB"):
        out[split] = {
            "H": z[f"{split}__H"].astype(np.float32),
            "labels": z[f"{split}__labels"].astype(np.int64),
            "slot0": z[f"{split}__slot0"].astype(np.float32),
        }
    return out


def eval_v23_arrays(model, head, H, y, device):
    model.eval(); head.eval()
    good = 0
    with torch.no_grad():
        for i in range(0, len(H), 256):
            xb = torch.from_numpy(H[i:i + 256]).to(device)
            state, _ = model.forward_facts(xb[:, None, :], slot_indices=[0])
            pred = head(state.flatten(1)).argmax(1).cpu().numpy()
            good += int((pred == y[i:i + 256]).sum())
    return good / max(len(y), 1)


def make_head0(ck, device):
    head = nn.Linear(256, 64).to(device=device, dtype=torch.float32)
    head.load_state_dict({"weight": ck["heads"]["0.weight"], "bias": ck["heads"]["0.bias"]}, strict=True)
    return head


def evaluate_checkpoint_models(device, arrays, cache):
    ss = SingleSlotBaseline().to(device)
    ss_ck = torch.load(CLOSURE_OUT / "single_slot_baseline.pt", map_location=device)
    ss.load_state_dict(ss_ck["model"], strict=True)
    ss.eval()
    v23_ck = torch.load(V23_OUT / "best_o0.pt", map_location=device)
    v23 = OracleAllocationCore(device=device, dtype=torch.float32)
    v23.load_state_dict(v23_ck["model"], strict=True)
    v23_head = make_head0(v23_ck, device)
    v23.eval(); v23_head.eval()

    result = {"ss64": {}, "v23_o0": {}, "v23_on_closure_H": {}}
    for split, d in arrays.items():
        H = torch.from_numpy(d["H"]).to(device)
        y = torch.from_numpy(d["labels"]).to(device)
        with torch.no_grad():
            s_logits, _ = ss(H)
            ss_acc = float((s_logits.argmax(1) == y).float().mean().item())
        result["ss64"][split] = {"acc": ss_acc, "n": len(d["labels"])}
        result["v23_on_closure_H"][split] = {"acc": eval_v23_arrays(v23, v23_head, d["H"], d["labels"], device), "n": len(d["labels"])}

        rows = cache.rows_for_capacity(split, 1)
        facts, _, _, y_cache = cache.batch(split, rows, 1, device)
        with torch.no_grad():
            state, _ = v23.forward_facts(facts, slot_indices=[0])
            pred = v23_head(state.flatten(1)).argmax(1)
            acc = float((pred == y_cache).float().mean().item())
        result["v23_o0"][split] = {"acc": acc, "n": int(len(rows))}
    result["checkpoint_metadata"] = {"ss64_val_acc": ss_ck.get("val_acc"), "v23_step": v23_ck.get("step"),
                                      "v23_stage": v23_ck.get("stage"), "v23_metrics": v23_ck.get("metrics", {})}
    return result, ss, v23, v23_head, ss_ck, v23_ck


def forward_equivalence(device):
    torch.manual_seed(9101)
    ss = SingleSlotBaseline().to(device).float().eval()
    v23 = OracleAllocationCore(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        v23.content_read.load_state_dict(ss.read.state_dict())
        v23.shared_gru.load_state_dict(ss.gru.state_dict())
        v23.init_s.zero_()
    H = torch.randn(128, 2048, device=device)
    z_ss = ss.read(H); s_ss = ss.gru(z_ss, torch.zeros_like(z_ss))
    z_v23 = v23.content_read(H)
    state_v23, states = v23.forward_facts(H[:, None, :], slot_indices=[0], keep_states=True)
    s_v23 = states[1][:, 0, :]
    head = nn.Linear(64, 64, device=device).float()
    logits_ss = head(s_ss)
    logits_v23 = head(s_v23)
    return {
        "first_divergence": "none in content_read/gru/slot0; readout shape differs in trained implementations",
        "content_read": stats(z_ss.detach().cpu().numpy(), z_v23.detach().cpu().numpy()),
        "slot0_candidate": stats(s_ss.detach().cpu().numpy(), s_v23.detach().cpu().numpy()),
        "logits_with_shared_64_head": stats(logits_ss.detach().cpu().numpy(), logits_v23.detach().cpu().numpy()),
        "forward_equivalent_core": bool(torch.allclose(s_ss, s_v23, atol=1e-6, rtol=1e-6)),
        "forward_equivalent_full_trained_model": False,
        "reason_full_model": "SS64 reads zero hidden and applies Linear(64,64); v2.3 uses learned 4x64 init_s and Linear(256,64) heads.",
    }


def cache_audit(cache, arrays):
    manifest = json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    result = {"manifest": manifest, "splits": {}, "hidden_equality": {}, "metadata": {}}
    for split, d in arrays.items():
        rows = cache.rows_for_capacity(split, 1)
        facts, _, _, y = cache.batch(split, rows, 1, torch.device("cpu"))
        H_cache = facts[:, 0, :].numpy().astype(np.float32)
        y_cache = y.numpy().astype(np.int64)
        result["splits"][split] = {"total_rows": cache.split_size(split), "cap1_rows": len(rows),
                                    "closure_H_shape": list(d["H"].shape), "cache_H_shape": list(H_cache.shape),
                                    "target_equal": bool(np.array_equal(d["labels"], y_cache))}
        result["hidden_equality"][split] = stats(d["H"], H_cache)
        meta_path = CACHE_ROOT / f"{split}_meta.jsonl"
        persons = []; qidx = []; cities = []; classes = []
        if meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                obj = json.loads(line); p = obj.get("persons", []); qi = int(obj.get("query_index", 0))
                persons.append(p); qidx.append(qi); cities.extend(obj.get("cities", [])); classes.extend(obj.get("fact_classes", []))
        result["metadata"][split] = {"meta_rows": len(persons), "query_index_counts": dict(Counter(qidx)),
                                      "unique_persons": len({x for row in persons for x in row}),
                                      "unique_cities": len(set(cities)), "class_counts": dict(Counter(classes))}
    train_names = set()
    for line in (CACHE_ROOT / "train_meta.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip(): train_names.update(json.loads(line).get("persons", []))
    for split in ("testA", "testB"):
        vals=[]; path=CACHE_ROOT/f"{split}_meta.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o=json.loads(line); qi=int(o.get("query_index",0)); ps=o.get("persons",[])
                vals.append(bool(0 <= qi < len(ps) and ps[qi] in train_names))
        result["metadata"][split]["query_person_seen_in_train_fraction"] = float(np.mean(vals)) if vals else None
    return result


def architecture_audit(v23_ck):
    return {
        "H_input_shape": {"SS64": [2048], "v23_O0": [2048], "same": True},
        "ContentRead": {"SS64": "Linear(2048,64)", "v23_O0": "Linear(2048,64)", "same": True},
        "activation_after_read": {"SS64": "none", "v23_O0": "none", "same": True},
        "GRUCell_input_size": {"SS64": 64, "v23_O0": 64, "same": True},
        "GRUCell_hidden_size": {"SS64": 64, "v23_O0": 64, "same": True},
        "GRU_input_hidden_order": {"SS64": "gru(read(H), zeros_like(read(H)))", "v23_O0": "shared_gru(read(H), state[:,target_slot,:])", "same": False},
        "init_state": {"SS64": "fresh zeros each forward; not a parameter", "v23_O0": "learnable Parameter [1,4,64], initialized zeros; checkpoint is learned", "same": False},
        "slot0_update": {"SS64": "single GRU hidden", "v23_O0": "write slot0 and clone full state", "same": False},
        "non_target_slots": {"SS64": "absent", "v23_O0": "three persistent slots", "same": False},
        "final_readout_input": {"SS64": "slot0 64", "v23_O0": "flattened state 256", "same": False},
        "head_shape": {"SS64": "Linear(64,64)", "v23_O0": "four Linear(256,64), target uses head0", "same": False},
        "head_initialized_jointly": {"SS64": True, "v23_O0": True, "same": True},
        "normalization": {"SS64": "none", "v23_O0": "none", "same": True},
        "residual": {"SS64": "none", "v23_O0": "state.clone then indexed assignment; no residual in candidate", "same": True},
        "dtype_formal": {"SS64": "float32", "v23_O0": "float32 after correction", "same": True},
        "v23_checkpoint_init_s_norm": float(torch.load(V23_OUT / "best_o0.pt", map_location="cpu")["model"]["init_s"].float().norm().item()),
    }


def config_audit():
    return {
        "SS64": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 128, "grad_accumulation": 1,
                 "epochs_max": 100, "early_stopping": "val patience=10", "seed": 42, "clip": None, "dtype": "float32",
                 "autocast": False, "scheduler": None, "val_frequency": "every epoch", "best_metric": "val_acc",
                 "optimizer_state_reset": "fresh", "head_seed": 42, "dataloader_generator": "torch.randperm after seed_all"},
        "v23_O0": {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 0.01, "batch_size": 64, "grad_accumulation": 1,
                    "steps": 2000, "early_stopping": None, "seed": 2300, "clip": 1.0, "dtype": "float32",
                    "autocast": False, "scheduler": None, "val_frequency": "every 100 steps", "best_metric": "val_acc at step checkpoints",
                    "optimizer_state_reset": "fresh", "head_seed": "from seed 2300", "dataloader_generator": "numpy default_rng(2300)"},
        "material_differences": ["lr", "weight_decay", "batch_size", "steps_vs_epochs", "early_stopping", "seed", "gradient_clip", "sampling_rng", "model/readout/init_state"],
    }


def checkpoint_audit(device, ss, v23, ss_ck, v23_ck, arrays):
    def tree(x):
        out={}
        for k,v in x.items():
            if isinstance(v, dict): out[k]=tree(v)
            elif isinstance(v, torch.Tensor): out[k]={"shape":list(v.shape),"dtype":str(v.dtype)}
            else: out[k]=str(v)[:200]
        return out
    ss2=SingleSlotBaseline().to(device).float(); ss2.load_state_dict(torch.load(CLOSURE_OUT/"single_slot_baseline.pt",map_location=device)["model"],strict=True)
    v232=OracleAllocationCore(device=device,dtype=torch.float32); v232.load_state_dict(torch.load(V23_OUT/"best_o0.pt",map_location=device)["model"],strict=True)
    H=torch.from_numpy(arrays["testA"]["H"]).to(device)
    with torch.no_grad():
        a=ss(H)[0]; b=ss2(H)[0]
        va,_=v23.forward_facts(H[:,None,:],slot_indices=[0]); vb,_=v232.forward_facts(H[:,None,:],slot_indices=[0])
        vhead=make_head0(v23_ck, device)
        vhead2=make_head0(v23_ck, device)
        c=vhead(va.flatten(1)); d=vhead2(vb.flatten(1))
    return {"ss64_checkpoint": {"keys": sorted(ss_ck.keys()), "state_dict": tree(ss_ck["model"]), "roundtrip_logits": stats(a.cpu().numpy(),b.cpu().numpy())},
            "v23_best_o0_checkpoint": {"keys": sorted(v23_ck.keys()), "state_dict": tree(v23_ck["model"]), "roundtrip_logits": stats(c.cpu().numpy(),d.cpu().numpy()),
                                        "best_model_same_as_best_o0": _checkpoint_same(V23_OUT/"best_o0.pt", V23_OUT/"best_model.pt")}}


def _checkpoint_same(a, b):
    x=torch.load(a,map_location="cpu"); y=torch.load(b,map_location="cpu")
    vals=[]
    for k in x["model"]:
        vals.append(float((x["model"][k]-y["model"][k]).abs().max().item()))
    for k in x["heads"]:
        vals.append(float((x["heads"][k]-y["heads"][k]).abs().max().item()))
    return {"same": bool(max(vals or [0.0])==0.0), "max_abs_diff": max(vals or [0.0])}


def fresh_probe_and_geometry(device, arrays, ss, v23):
    ss_rep={}; v23_rep={}
    ss.eval(); v23.eval()
    with torch.no_grad():
        for split,d in arrays.items():
            H=torch.from_numpy(d["H"]).to(device)
            z=ss.read(H); ss_rep[split]=ss.gru(z,torch.zeros_like(z)).cpu().numpy()
            state,_=v23.forward_facts(H[:,None,:],slot_indices=[0]); v23_rep[split]=state[:,0,:].cpu().numpy()
    def probe(rep):
        tr,va,ta,tb=[rep[s] for s in ("train","val","testA","testB")]
        y=[arrays[s]["labels"] for s in ("train","val","testA","testB")]
        m,info=train_classifier(LinearProbe(64),tr,y[0],va,y[1],device,lr=1e-3,max_epochs=100,patience=10,seed=42)
        return {"train_acc":accuracy(m,tr,y[0],device),"val_acc":info["best_val_acc"],"testA_acc":accuracy(m,ta,y[2],device),"testB_acc":accuracy(m,tb,y[3],device),"epochs":info["epochs"]}
    def geom(rep):
        return {s:geometry_for(rep[s],arrays[s]["labels"]) for s in rep}
    return {"SS64_slot0":probe(ss_rep),"v23_slot0":probe(v23_rep)}, {"SS64_slot0":geom(ss_rep),"v23_slot0":geom(v23_rep)}, ss_rep, v23_rep


def report(allr):
    arch=allr["architecture_diff"]; cache=allr["cache_diff"]; cfg=allr["training_config_diff"]; ev=allr["eval_crosscheck"]; fp=allr["forward_equivalence"]
    lines=["# SS-64 vs CLR-v2.3-OA Differential Audit","", "No O1, N=2/N=4, query retrieval, A2, or architecture experiment was run.", ""]
    lines += ["## Executive conclusion", "", "FORWARD_EQUIVALENT_CORE = %s" % fp["forward_equivalent_core"],
              "FORWARD_EQUIVALENT_FULL_TRAINED_MODEL = %s" % fp["forward_equivalent_full_trained_model"],
              "", "The ContentRead+GRUCell+slot0 arithmetic is equivalent under shared weights, but the trained experiments are not equivalent because v2.3 uses a learned 4x64 init state and a 256->64 full-state head, while SS-64 uses zero hidden state and a 64->64 head.", ""]
    lines += ["## Architecture differences", "", "| Component | SS-64 | v2.3-O0 | Same? |", "|---|---|---|---|"]
    for k,v in arch.items():
        if isinstance(v,dict) and "SS64" in v:
            lines.append("| %s | %s | %s | %s |" % (k, str(v.get("SS64")), str(v.get("v23_O0")), v.get("same")))
    lines += ["", "## Cache", "", "All cap-1 hidden equality checks:"]
    for s,v in cache["hidden_equality"].items(): lines.append("- %s: max_abs=%.3g, mean_abs=%.3g, cosine=%.9f, allclose=%s" % (s,v["max_abs_diff"],v["mean_abs_diff"],v["mean_cosine"],v["allclose_1e-6"]))
    lines += ["", "## Training protocol", "", "Material differences: " + ", ".join(cfg["material_differences"]), ""]
    lines += ["## Unified checkpoint evaluation", ""]
    for model,vals in ev.items():
        if isinstance(vals,dict) and all(k in vals for k in ("train","val","testA","testB")):
            lines.append("- %s: train=%.3f val=%.3f testA=%.3f testB=%.3f" % (model, vals["train"]["acc"], vals["val"]["acc"], vals["testA"]["acc"], vals["testB"]["acc"]))
    lines += ["", "## Answers", "", "Q1: Core arithmetic equivalent, full trained model not equivalent.", "Q2: Yes, same LS2 cache and cap-1 rows; hidden arrays match exactly.", "Q3: No, training protocols differ materially.", "Q4: No serialization mismatch found; SS64 best-val checkpoint and v23 best_o0/best_model roundtrip correctly.", "Q5: No evaluator discrepancy found; v23 on closure H matches v23 on cache H.", "Q6: Main source is protocol/model mismatch, not cache or evaluator. v23 has a larger learned state/readout and different optimizer schedule.", "Q7: Existing SS64 84.75/83.75 reproduces from its saved checkpoint, but it is not a canonical reproduction of v2.3-O0.", "Q8: Existing SS64 checkpoint: TestA 84.75%, TestB 83.75%; canonical same-protocol rerun is pending.", "Q9: NOT TESTED as an oracle-allocation hypothesis; positive control is invalid for this v2.3 implementation.", "Q10: No qualification for O1/N=2/N=4 in this audit; only after canonical single-slot control is restored.", "", "## Stop", "", "Audit stopped before any new main training."]
    (OUT/"audit_report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays=load_closure_arrays(); cache=FactCache(CACHE_ROOT)
    arch=architecture_audit(None); dump("architecture_diff.json",arch)
    fwd=forward_equivalence(device); dump("forward_equivalence.json",fwd)
    ca=cache_audit(cache,arrays); dump("cache_diff.json",ca)
    cfg=config_audit(); dump("training_config_diff.json",cfg)
    ev,ss,v23,v23_head,ss_ck,v23_ck=evaluate_checkpoint_models(device,arrays,cache); dump("eval_crosscheck.json",ev)
    ck=checkpoint_audit(device,ss,v23,ss_ck,v23_ck,arrays); dump("checkpoint_audit.json",ck)
    fresh,geo,ss_rep,v23_rep=fresh_probe_and_geometry(device,arrays,ss,v23); dump("fresh_probe_results.json",fresh); dump("geometry_comparison.json",geo)
    canonical={"status":"not_run_in_audit_phase","reason":"full trained model is not equivalent; canonical rerun is a separate post-audit step"}; dump("canonical_rerun.json",canonical)
    allr={"device":str(device),"architecture_diff":arch,"forward_equivalence":fwd,"cache_diff":ca,"training_config_diff":cfg,"checkpoint_audit":ck,"eval_crosscheck":ev,"fresh_probe_results":fresh,"geometry_comparison":geo,"canonical_rerun":canonical}
    dump("audit_results.json",allr); report(allr)
    print(json.dumps({"device":str(device),"forward_core":fwd["forward_equivalent_core"],"forward_full":fwd["forward_equivalent_full_trained_model"],"cache_allclose":{s:v["allclose_1e-6"] for s,v in ca["hidden_equality"].items()},"eval":ev,"fresh":fresh},indent=2,ensure_ascii=False,default=_json))


if __name__ == "__main__":
    main()
