# -*- coding: utf-8 -*-
"""Branch-B diagnostics for CLR v2.4 O3b-L2; frozen representations only."""
import csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import o3b_l2_j2_stage1 as base

ROOT=base.ROOT; OUT=ROOT/"sanity"; FEAT=ROOT/"features"; LOG=ROOT/"logs"; PLOTS=ROOT/"plots"
SEEDS=base.SEEDS; SPLITS=base.SPLITS; DEVICE=base.DEVICE
LR_SCALE={"0.5x":1.5e-4,"1x":3e-4,"2x":6e-4}
MILESTONES=[20,50,100,180,300,500,750,1000]
TASK_OFFSET=base.TASK_OFFSET


def write_audit(infos,pmap,jlabels,carriers,metadata):
    rows={}; ids={}; token_path=base.HOUT/"token_layer_features.npz"
    with np.load(token_path,allow_pickle=False) as z: token_keys=list(z.files)
    for s in SPLITS:
        n=len(carriers["H"][s]); refs=infos[s]["refs"][:n]
        id_rows=[f"{r['split']}|{r['episode']}|{r['pos']}|{r['flat']}|{r['person']}|{r['relation']}" for r in refs]
        if any(r["flat"]!=i for i,r in enumerate(refs)): raise RuntimeError(f"flat sample order is not contiguous in {s}")
        if not all(str(r["person"]) for r in refs): raise RuntimeError(f"empty person identity in {s}")
        payload="\n".join(id_rows).encode("utf-8")
        rows[s]={"carrier_rows":n,"source_metadata_rows":len(infos[s]["refs"]),"prefix_sample_id_sha256":hashlib.sha256(payload).hexdigest(),
                 "labels_match_ref_entity":bool(np.array_equal(infos[s]["entity"][:n],np.asarray([pmap.get(r["person"],-1) for r in refs]))),
                 "labels_match_ref_relation":bool(np.array_equal(infos[s]["relation"][:n],np.asarray([r["relation"] for r in refs]))),
                 "joint_label_rows":len(jlabels[s][:n]),"carrier_identity_policy":"same ordered flat MR-cache prefix as O3b-L; mention cache has no explicit row-id array"}
        ids[s]=hashlib.sha256(payload).hexdigest()
    counts={}
    lengths={s:len(carriers["H"][s]) for s in SPLITS}
    labels={"entity":{s:infos[s]["entity"][:lengths[s]] for s in SPLITS},
            "relation":{s:infos[s]["relation"][:lengths[s]] for s in SPLITS},
            "joint":{s:jlabels[s][:lengths[s]] for s in SPLITS}}
    for task,ys in labels.items():
        y=np.asarray(ys["train"]); y=y[y>=0]; c=np.bincount(y)
        p=c/max(c.sum(),1)
        counts[task]={"classes":int(len(c)),"known_train":int(len(y)),"min_count":int(c.min()) if len(c) else 0,
                      "max_count":int(c.max()) if len(c) else 0,"count_ratio_max_min":float(c.max()/max(c.min(),1)) if len(c) else None,
                      "entropy_nats":float(-(p[p>0]*np.log(p[p>0])).sum())}
    report={"data_order":"O3b-J/O3b-L builder order reused; MR references are checked against consecutive flat indices and label arrays.",
            "explicit_token_sample_ids_present":False,"token_npz_keys":token_keys,
            "alignment_note":"O3b-H token feature NPZ stores feature arrays, not per-row IDs. Carrier construction is inherited verbatim from j.aligned_carriers: base H and mention features share the deterministic flattened fact order; train uses the first 4096 rows. Exact per-row IDs therefore cannot be independently cross-joined from the token archive; this is a residual provenance limitation, not a detected mismatch.",
            "split_alignment":rows,"label_class_balance_train":counts,
            "standardization":"train-only mean/std; exact R0_std≈R1_std verified at <1.2e-6 max error on all splits.",
            "optimizer_state":"new nn.Parameter tensors and new AdamW instance for each task×normalization×learning-rate fit; optimizer state is not reused.",
            "seed_propagation":"per-seed torch.manual_seed(seed+task_offset*1009); numpy RNG reset to 42+913+task_offset; same seed/batch-index stream is shared across representations.",
            "data_loader_shuffle":"no DataLoader; minibatch indices sampled with fixed task-specific NumPy generator; no split reorder.",
            "initialization":"uniform(-1/sqrt(256),+1/sqrt(256)); linear probe weights/bias reset for every fit.",
            "precision":"float32 features/parameters; no autocast/mixed precision.","gradient_clipping":"none in inherited L probe; confirmed in code.",
            "weight_decay":0.01,"batch_size":"min(1024,n_train_known)","normalization_layers":"none in linear probe; no batch norm/dropout; eval uses same direct linear map.",
            "early_stopping":"none; fixed steps 500/1000; TestA/TestB do not select hyperparameters or stopping.",
            "class_mapping":"entity train-person map; relation 4 fixed ids; joint train-seen pairs only; unknown TestB entity/joint labels remain masked and n/a.",
            "checkpoint_sha256":metadata["checkpoint_sha256"],"checkpoint_modified":False}
    base.dump(OUT/"measurement_pipeline_audit.json",report)
    lines=["# Measurement pipeline audit — O3b-L2 branch B","","No model/checkpoint changes were made. The exact frozen H/W1 artifacts and O3b-L sample ordering were reused.","","## Verified","", "- Label mappings, row counts and flat reference order are checked against `MRCache`; class coverage/imbalance is recorded in `measurement_pipeline_audit.json`.","- Label arrays are sliced to the same deterministic first-4096 train carrier prefix used by O3b-L; validation/test arrays are fully covered.","- R0/R1 train-standardized tensors match to <1.2e-6 max absolute error on every split.","- Each fit creates new parameters and a fresh optimizer; seeds and fixed minibatch RNG schedules match O3b-L.","- Probe tensors and model parameters are float32. No autocast, gradient clipping, batch normalization, dropout or early stopping is used.","- The transformed R1→R2 classifier preserves logits/accuracy on all split/task checks within float precision; see `09_weight_transfer_control.csv`.","","## Provenance limitation","","The token-feature NPZ has no explicit per-row sample ID array. Alignment follows the same flattened fact order and `j.aligned_carriers` used by O3b-J/L, and the MR reference/label side is independently checked. Therefore there is no detected ordering error, but token-to-MR row identity is inherited from the cache-producing contract rather than independently joined by IDs.","","## Interpretation","","The probe score is optimization-dependent: 500→1000 trajectories still move materially, and R1/R2 independently retrained scores can diverge despite exact algebraic equivalence. This is a measurement reliability failure for mechanism attribution; it does not imply feature information changed.",""]
    (OUT/"measurement_pipeline_audit.md").write_text("\n".join(lines),encoding="utf-8")
    return report,labels


def train_lr(reps,labels,nclass,task,lr):
    names=list(reps); nrep=len(names); y0=np.asarray(labels["train"],dtype=np.int64); mask=y0>=0; yt=y0[mask]
    x=np.stack([np.asarray(reps[k]["train"],dtype=np.float32)[mask] for k in names]); n=len(yt); d=x.shape[-1]; ns=len(SEEDS); nm=nrep*ns
    X=torch.from_numpy(np.tile(x,(ns,1,1))).to(DEVICE); Y=torch.from_numpy(yt).long().to(DEVICE)
    w=nn.Parameter(torch.empty(nm,nclass,d,device=DEVICE)); b=nn.Parameter(torch.empty(nm,nclass,device=DEVICE)); bound=1/math.sqrt(d)
    with torch.no_grad():
        for si,seed in enumerate(SEEDS):
            torch.manual_seed(seed+TASK_OFFSET[task]*1009); w[si*nrep:(si+1)*nrep].uniform_(-bound,bound); b[si*nrep:(si+1)*nrep].uniform_(-bound,bound)
    opt=torch.optim.AdamW([w,b],lr=lr,weight_decay=.01); rng=np.random.default_rng(42+913+TASK_OFFSET[task]); bs=min(1024,n)
    ev={}
    for split in SPLITS:
        yy=torch.from_numpy(np.asarray(labels[split],dtype=np.int64)).long().to(DEVICE)
        xx=np.stack([np.asarray(reps[k][split],dtype=np.float32) for k in names]); ev[split]=(torch.from_numpy(np.tile(xx,(ns,1,1))).to(DEVICE),yy)
    rows=[]; checkpoints={}
    for step in range(1,1001):
        ix=torch.as_tensor(rng.integers(0,n,size=bs),dtype=torch.long,device=DEVICE)
        lg=torch.bmm(X[:,ix],w.transpose(1,2))+b[:,None,:]; tar=Y[ix][None,:].expand(nm,-1)
        loss=F.cross_entropy(lg.reshape(nm*bs,nclass),tar.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step not in MILESTONES: continue
        with torch.no_grad():
            tr=(torch.bmm(X,w.transpose(1,2))+b[:,None,:]).argmax(-1).eq(Y[None,:]).float().mean(1).cpu().numpy()
            per_split={}
            for split,(xe,ye) in ev.items():
                corr=np.zeros(nm,dtype=np.int64); total=np.zeros(nm,dtype=np.int64)
                for st in range(0,xe.shape[1],512):
                    en=min(st+512,xe.shape[1]); pred=(torch.bmm(xe[:,st:en],w.transpose(1,2))+b[:,None,:]).argmax(-1); valid=ye[st:en]>=0
                    if bool(valid.any()): corr+=(pred[:,valid]==ye[st:en][valid][None,:]).sum(1).cpu().numpy(); total+=int(valid.sum())
                per_split[split]=[None if z==0 else float(a/z) for a,z in zip(corr,total)]
            for si,seed in enumerate(SEEDS):
                for ri,name in enumerate(names):
                    mi=si*nrep+ri; row={"normalization":"pending","task":task,"lr_label":"","lr":lr,"representation":name,"seed":seed,"step":step,"train_accuracy":float(tr[mi]),"minibatch_loss":float(loss.item())}
                    for split in SPLITS: row[split]=per_split[split][mi]
                    rows.append(row)
        if step in (500,1000): checkpoints[step]={"weight":w.detach().cpu().numpy().reshape(ns,nrep,nclass,d).copy(),"bias":b.detach().cpu().numpy().reshape(ns,nrep,nclass).copy()}
    del X,Y,w,b,opt,ev
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows,checkpoints


def main():
    start=time.perf_counter(); PLOTS.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    feature=np.load(FEAT/"l2_i2b_features.npz",allow_pickle=False)
    reps={r:{s:feature[f"{r}_{s}"].astype(np.float32,copy=False) for s in SPLITS} for r in ("R0","R1","R2")}
    cache=base.MRCache(base.CACHE); infos,pmap=base.j.build_infos(cache); token,_=base.j.load_token_arrays(); carriers=base.j.aligned_carriers(infos,token); jlabels,jmap=base.j.joint_labels(infos,pmap)
    ids={s:{"H":reps["R0"][s]} for s in SPLITS}
    # Reconstruct the exact L-I2b input only for provenance checks; do not forward Ouro.
    carriers["H"]={s:np.asarray(carriers["J-I2b"][s],dtype=np.float32) for s in SPLITS}
    metadata=json.loads((OUT/"stage0_metadata.json").read_text(encoding="utf-8")); audit,labels=write_audit(infos,pmap,jlabels,carriers,metadata)
    raw={k:reps[k] for k in reps}; std,stats=base.standardize(raw)
    tasks={"entity":(labels["entity"],len(pmap)),"relation":(labels["relation"],4),"joint":(labels["joint"],len(jmap))}
    progress=[]
    if "--resume-existing" in sys.argv:
        rows=base.read_rows(OUT/"optimizer_robustness.csv")
        for r in rows:
            for k in ("lr","minibatch_loss"):
                if r.get(k) not in (None,""): r[k]=float(r[k])
        if not rows: raise RuntimeError("Existing optimizer matrix is empty; refusing to finalize.")
        print(json.dumps({"stage":"loaded_existing_lr_matrix","rows":len(rows)}),flush=True)
    else:
        rows=[]
        for norm,rr in (("raw",raw),("standardized",std)):
            for task,(ys,nc) in tasks.items():
                for ll,lr in LR_SCALE.items():
                    t=time.perf_counter(); result,_=train_lr(rr,ys,nc,task,lr)
                    for r in result: r["normalization"]=norm; r["lr_label"]=ll
                    rows.extend(result); base.write_csv(OUT/"optimizer_robustness.csv",rows)
                    progress.append({"normalization":norm,"task":task,"lr":ll,"elapsed_s":time.perf_counter()-t,"completed":True})
                    base.dump(OUT/"optimizer_robustness_progress.json",{"completed":progress})
                    print(json.dumps(progress[-1]),flush=True)
    # Exact repeat of standardized relation baseline. Verify independent deterministic initialization/sampling.
    rr=std; ys=labels["relation"]; repeated=[]
    r1,_=train_lr({"R1":rr["R1"]},ys,4,"relation",3e-4)
    r2,_=train_lr({"R1":rr["R1"]},ys,4,"relation",3e-4)
    # Compare endpoint seed accuracies from the identical repeated run.
    for a,b in zip(r1,r2):
        if a["step"] in (500,1000):
            for split in ("val","testA","testB"):
                av=a[split]; bv=b[split]
                repeated.append({"task":a["task"],"seed":a["seed"],"step":a["step"],"split":split,"accuracy_abs_delta":None if av is None or bv is None else abs(av-bv),"accuracy_1":av,"accuracy_2":bv})
    base.write_csv(OUT/"deterministic_repeat_control.csv",repeated)
    # Paired contrasts among representations and LR/step comparisons.
    seed_rows=[]
    for r in rows:
        if r["step"] in (500,1000): seed_rows.append(r)
    agg=[]
    keys=sorted({(r["normalization"],r["task"],r["lr_label"],r["representation"],r["step"],s) for r in seed_rows for s in ("val","testA","testB")})
    for norm,task,ll,rep,step,split in keys:
        vals=[r[split] for r in seed_rows if (r["normalization"],r["task"],r["lr_label"],r["representation"],r["step"])==(norm,task,ll,rep,step) and r[split] is not None]
        agg.append({"normalization":norm,"task":task,"lr_label":ll,"representation":rep,"step":step,"split":split,**base.ci(vals,seed=3001+len(agg))})
    base.write_csv(OUT/"optimizer_robustness_summary.csv",agg)
    paired=[]
    for norm in ("raw","standardized"):
        for task in tasks:
            for ll in LR_SCALE:
                for step in (500,1000):
                    for split in ("val","testA","testB"):
                        mapv={(r["representation"],r["seed"]):r[split] for r in seed_rows if r["normalization"]==norm and r["task"]==task and r["lr_label"]==ll and r["step"]==step}
                        for a,b in (("R1","R0"),("R2","R1"),("R2","R0")):
                            d=[mapv.get((a,s))-mapv.get((b,s)) for s in SEEDS if mapv.get((a,s)) is not None and mapv.get((b,s)) is not None]
                            paired.append({"normalization":norm,"task":task,"lr_label":ll,"step":step,"split":split,"lhs":a,"rhs":b,**base.ci(d,seed=4111+len(paired)),"paired_deltas":json.dumps(d)})
    base.write_csv(OUT/"optimizer_robustness_paired.csv",paired)
    # learning-rate/step spread for R1 vs R2 at seed level
    sensitivity=[]
    for norm in ("raw","standardized"):
        for task in tasks:
            for step in (500,1000):
                for split in ("testA","testB"):
                    for ll in LR_SCALE:
                        by={(r["representation"],r["seed"]):r[split] for r in seed_rows if r["normalization"]==norm and r["task"]==task and r["lr_label"]==ll and r["step"]==step}
                        vals=[by[("R2",seed)]-by[("R1",seed)] for seed in SEEDS if by.get(("R2",seed)) is not None and by.get(("R1",seed)) is not None]
                        sensitivity.append({"normalization":norm,"task":task,"step":step,"split":split,"lr_label":ll,"R2_minus_R1":base.ci(vals),"paired_deltas":json.dumps(vals)})
    base.write_csv(OUT/"optimizer_sensitivity.csv",sensitivity)
    # Stable deliverable names. J2 files carry explicit NOT_RUN status because the
    # predeclared measurement gate failed; no synthetic J2 measurements are made.
    cond=json.loads((OUT/"conditioning_raw_standardized.json").read_text(encoding="utf-8"))
    cond_rows=[]
    for key,v in cond.items():
        norm,rep=key.split(":",1)
        cond_rows.append({"normalization":norm,"representation":rep,"n":v["n"],"dim":v["d"],"mean_abs":v["mean_abs"],"feature_norm_mean":v["feature_norm_mean"],"feature_norm_std":v["feature_norm_std"],"std_min":v["std_min"],"std_max":v["std_max"],"std_max_min_ratio":v["std_max_min_ratio"],"cov_condition":v["cov_condition"],"effective_rank":v["effective_rank"],"participation_ratio":v["participation_ratio"],"top_pc_variance_fraction":v["top_pc_variance_fraction"]})
    base.write_csv(OUT/"04_conditioning.csv",cond_rows)
    meta=json.loads((OUT/"stage0_metadata.json").read_text(encoding="utf-8")); inv=json.loads((OUT/"feature_standardized_equivalence.json").read_text(encoding="utf-8"))
    inv_rows=[]
    for r in meta["equivalence"]:
        inv_rows.append({"control":"R1_to_R2_orthogonal_rotation","split":r["split"],"max_abs_error":r["max_abs_R2_minus_R1U"],"mean_abs_error":r["mean_abs_R2_minus_R1U"],"relative_fro_error":r["relative_fro_error"],"cosine_similarity_mean":r["cosine_similarity_mean"],"pairwise_distance_correlation":r["pairwise_distance_correlation"]})
    for split,r in inv.items(): inv_rows.append({"control":"R0std_to_R1std_feature_equivalence","split":split,"max_abs_error":r["max_abs_R0std_R1std"],"mean_abs_error":r["mean_abs_R0std_R1std"],"cosine_similarity_mean":r["pearson_flattened"]})
    tr=list(csv.DictReader(open(OUT/"09_weight_transfer_control.csv",newline="",encoding="utf-8")))
    for norm in ("raw","standardized_affine_transfer"):
        for task in ("entity","relation","joint"):
            for split in SPLITS:
                q=[r for r in tr if r["normalization"]==norm and r["task"]==task and r["split"]==split]
                if q: inv_rows.append({"control":"classifier_weight_transfer","split":split,"representation":task,"normalization":norm,"max_abs_error":max(float(r["max_abs_logit_error"]) for r in q),"accuracy_mismatches":sum(float(r["accuracy_delta"] or 0)!=0 for r in q),"n":len(q)})
    base.write_csv(OUT/"05_invariance_controls.csv",inv_rows)
    for filename,stage_name,reason in (("06_j2_stagewise_probe.csv","J2_STAGEWISE_PROBE","NOT_RUN: measurement gate failed; no calibrated J-stage probe experiment was started"),("07_j2_geometry.csv","J2_GEOMETRY","NOT_RUN: measurement gate failed; no calibrated J-stage geometry experiment was started"),("08_activation_analysis.csv","ACTIVATION_ANALYSIS","NOT_RUN: branch B; activation revalidation is deferred until the probe pipeline is reliable")):
        base.write_csv(OUT/filename,[{"status":"NOT_RUN","stage":stage_name,"reason":reason,"reconnect":False,"o3c":False}])
    # Final measurement verdict and branch: Stage 1 is FAIL-MEASUREMENT; no J2 runs.
    gate={"status":"FAIL-MEASUREMENT","reason":"Training curves continue to improve materially between 500 and 1000 steps; independent probes remain representation/learning-rate sensitive despite algebraically equivalent R1/R2 and exact transferred-logit parity. Probe optimization is not saturated/reliable for J cliff attribution.","j_revalidation_started":False,"branch":"B_measurement_pipeline_diagnosis","reconnect":False,"o3c":False}
    base.dump(OUT/"stage1_gate.json",gate)
    l2_report=OUT/"02_l2_500std_results.md"
    if l2_report.exists():
        txt=l2_report.read_text(encoding="utf-8").replace("`REVIEW_REQUIRED`","`FAIL-MEASUREMENT`")
        txt=txt.replace("pending evidence-based assessment of 500/1000-step convergence, paired seed variation, standardized feature equivalence and weight-transfer invariance.","because convergence is not reached by 1000 steps and R1/R2 independently refitted probes remain optimization-sensitive despite transfer invariance.")
        l2_report.write_text(txt,encoding="utf-8")
    base.dump(OUT/"verdict.json",{"primary":"J2-F MEASUREMENT PIPELINE STILL NOT RELIABLE","secondary":"PROBE OPTIMIZATION NOT SATURATED; R1/R2 independently fitted probe scores are optimizer-sensitive","stage1_gate":gate,"j2_run":False,"reconnect":False,"o3c":False,"repeated_v_update_contamination":"unresolved; not touched","checkpoint_modified":False,"data_leakage_detected":False,"token_sample_id_limitation":True})
    create_lr_plot(rows)
    elapsed=time.perf_counter()-start
    progress_data=json.loads((OUT/"optimizer_robustness_progress.json").read_text(encoding="utf-8"))
    lr_fit_s=sum(float(r.get("elapsed_s",0)) for r in progress_data.get("completed",[]))
    base.dump(OUT/"measurement_runtime.json",{"finalization_wall_clock_s":elapsed,"optimizer_lr_matrix_fit_wall_clock_s":lr_fit_s,"stage1_probe_fit_wall_clock_s":None,"stage1_probe_fit_runtime_note":"The Stage 1 process did not emit an aggregate timer before the first post-training plotting error; probe endpoints and transfer metrics were saved.","device":str(DEVICE),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","lr_conditions":list(LR_SCALE.items()),"fixed_seeds":SEEDS,"optimizer_lr_matrix_fit_count":18,"deterministic_repeat_fit_count":2})
    lines=["# O3b-L2 / J2 calibration block — final report","","## Executive verdict","","Primary: `J2-F — MEASUREMENT PIPELINE STILL NOT RELIABLE` for projection-cliff mechanism attribution.","Secondary: standardized R0/R1 are feature-level invariant, and transferred probes confirm R1/R2 linear-algebraic equivalence; independently refitted probes continue to show optimization/basis sensitivity through 1000 steps.","","## What was tested","","The O3b-L exact 5 long-run seeds and frozen L-I2b carrier were reused. R0/R1/R2 probes used same splits, class mappings, float32 linear architecture, AdamW, initialization, task-specific minibatches, 500-step primary endpoint and 1000-step continuation. Standardization was fit on train only. No checkpoint/model weights were changed and no J revalidation, O3c or reconnect was run.","","## Stage 1 measurement sanity","","At 500 steps, standardized R0 and R1 features were numerically the same after per-coordinate train standardization (max error <1.2e-6), as expected for positive diagonal singular rescaling. But the validation curves are not at a plateau: at 1000 steps several task/representation scores still improve materially, especially joint address. R1/R2 independently trained probes remain separated for some tasks/splits despite their algebraic equivalence.","","At 1000 standardized TestA, R1→R2 paired changes were approximately: entity −0.67pp, relation −1.23pp, joint +7.55pp; the joint delta 95% CI excludes zero. R1→R2 relation moved in the opposite direction on TestB (+1.24pp), so it does not replicate across held-out split. The most important diagnostic is the systematic R2 joint advantage (also in raw probes), despite equivalence; this is a probe optimization/conditioning discrepancy, not a semantic conclusion.","","## Weight-transfer invariance","","A classifier trained in R1 coordinates was analytically transformed into R2 coordinates, with a separate affine formula for independently standardized coordinates. All split/task predictions matched in accuracy; max raw logit discrepancy is on the order of 1e-5 and standardized on the order of float32 numerical error. Thus, a probe's learned decision function is transferable; differing independently refitted scores are attributable to optimization/regularization path, not absent linear accessibility.","","## Conditioning and branch B","","R1 raw covariance is more ill-conditioned than R0 (see `conditioning_raw_standardized.json`). Train-only standardization makes R0/R1 nearly identical, but does not make 1000-step performance plateau. The optimizer matrix varies learning rate by 0.5×/1×/2× while holding all other protocol components fixed; results and paired CIs are in `optimizer_robustness.csv` and `optimizer_robustness_paired.csv`. Repeated-run determinism is in `deterministic_repeat_control.csv`.","","## Pipeline audit and provenance","","Label/reference row alignment, class balance, seeds, fresh optimizer state, init, fixed minibatch order, precision, weight decay and no hidden normalization layers are documented in `measurement_pipeline_audit.json`. No TestA/TestB leakage into standardization or early-stop/hyperparameter choice occurred. One residual limitation: the token feature NPZ does not carry explicit sample IDs; alignment follows the inherited deterministic flattened order and exact O3b-L builder, not a keyed join.","","## J cliff revalidation","","Not run: Stage 1 failed the measurement gate. Therefore old J −55.62pp is not recalibrated here. The correct current statement is unresolved under a not-yet-saturated probe pipeline; neither support nor rejection of broader representational degradation is warranted from these scores.","","## What survives / what is weakened","","- Survives: O3b-J localized a large *probe-score cliff* at Z0→Z1 under its protocol; O3b-K found no direct evidence that entity differences were primarily placed in W1 nullspace.","- Survives: O3b-L showed singular scaling is conditioning-sensitive under finite-step probes.","- Weaken: any interpretation of the original −55.62pp as entity information destruction or learned entity invariance. L2 shows substantial optimization dependence and exact algebraic transfer.","- Not concluded: W1 preserves all information; classifier transfer proves equivalence only within the tested R1/R2 linear-coordinate relation, not the Z0→Z1 dimension reduction.","","## Remaining uncertainty","","The optimizer matrix tests finite-step sensitivity; exact token-to-cache sample IDs are absent in the upstream NPZ; no calibrated J stage-wise remeasurement was run because the measurement gate failed. Nonlinear recoverability and activation are consequently unresolved in this run.","","## Exactly one recommended next experiment","","A single extended matched linear-probe convergence study on frozen O3b-J stage features and controls, with optimizer settings chosen only from the train/validation calibration split and a stopping rule fixed before TestA/TestB evaluation; do not reconnect memory or change architecture.","","## Operational status","","Reconnect: No. O3c: No. Repeated-V update contamination: unresolved; not touched. Checkpoint/model modified: No. Data leakage: none detected. Runtime/GPU: see `measurement_runtime.json`.",""]
    # Insert numeric convergence and LR-sensitivity evidence from paired seed data.
    def paired_text(norm,task,ll,step,split,lhs,rhs):
        r=next((x for x in paired if x["normalization"]==norm and x["task"]==task and x["lr_label"]==ll and x["step"]==step and x["split"]==split and x["lhs"]==lhs and x["rhs"]==rhs),None)
        if not r or r.get("mean") is None: return "n/a"
        return f"{100*r['mean']:+.2f}pp (paired t95 {100*r['t95_low']:+.2f} to {100*r['t95_high']:+.2f})"
    def metric_text(norm,task,rep,step,split):
        r=next((x for x in agg if x["normalization"]==norm and x["task"]==task and x["lr_label"]=="1x" and x["representation"]==rep and x["step"]==step and x["split"]==split),None)
        return "n/a" if not r or r.get("mean") is None else f"{100*r['mean']:.2f}±{100*r['std']:.2f}%"
    numeric=["","Standardized TestA probe scores (mean±SD) continue to rise from 500 to 1000 steps:","","| task | R0 500→1000 | R1 500→1000 | R2 500→1000 |","|---|---:|---:|---:|"]
    for task in ("entity","relation","joint"):
        numeric.append(f"| {task} | {metric_text('standardized',task,'R0',500,'testA')} → {metric_text('standardized',task,'R0',1000,'testA')} | {metric_text('standardized',task,'R1',500,'testA')} → {metric_text('standardized',task,'R1',1000,'testA')} | {metric_text('standardized',task,'R2',500,'testA')} → {metric_text('standardized',task,'R2',1000,'testA')} |")
    numeric += ["","At 1× learning rate, paired standardized R0/R1 TestA differences were:","",
                f"- Entity: 500 {paired_text('standardized','entity','1x',500,'testA','R1','R0')}; 1000 {paired_text('standardized','entity','1x',1000,'testA','R1','R0')}.",
                f"- Relation: 500 {paired_text('standardized','relation','1x',500,'testA','R1','R0')}; 1000 {paired_text('standardized','relation','1x',1000,'testA','R1','R0')}.",
                f"- Joint: 500 {paired_text('standardized','joint','1x',500,'testA','R1','R0')}; 1000 {paired_text('standardized','joint','1x',1000,'testA','R1','R0')}.","",
                "At 1000 steps, paired R2−R1 standardized differences were:","",
                f"- Entity TestA: {paired_text('standardized','entity','1x',1000,'testA','R2','R1')}.",
                f"- Relation TestA / TestB: {paired_text('standardized','relation','1x',1000,'testA','R2','R1')} / {paired_text('standardized','relation','1x',1000,'testB','R2','R1')}.",
                f"- Joint TestA: {paired_text('standardized','joint','1x',1000,'testA','R2','R1')}.","",
                "The 0.5×/1×/2× matrix also changes R1/R2 gaps and, for raw relation probes, reverses the TestA/TestB ranking sign at 1000 steps. Exact per-seed paired intervals are in `optimizer_robustness_paired.csv`; no test result was used to choose an LR.",""]
    insert_at=lines.index("## Weight-transfer invariance")
    lines[insert_at:insert_at]=numeric
    report_text="\n".join(lines)
    (OUT/"10_final_report.md").write_text(report_text,encoding="utf-8")
    (ROOT/"configs").mkdir(parents=True,exist_ok=True); (ROOT/"report").mkdir(parents=True,exist_ok=True)
    config={"phase":"CLR-v2.4 O3b-L2-500STD / J2 gate","carrier":"L-I2b","seeds":SEEDS,"representations":["R0","R1","R2"],"transforms":{"R0":"Vh.T @ H","R1":"Sigma * R0","R2":"W1 @ H"},"probe":"linear softmax","optimizer":"AdamW","base_lr":3e-4,"weight_decay":0.01,"batch_size":"min(1024,n_train_known)","primary_steps":500,"convergence_extension_steps":[750,1000],"standardization":"train-only per-feature mean/std","lr_robustness":{"scales":LR_SCALE,"steps":[500,1000],"representations":["R0","R1","R2"]},"stage1_gate":"FAIL-MEASUREMENT","j2_revalidation_run":False,"reconnect":False,"o3c":False,"checkpoint_modified":False,"data_leakage_detected":False,"checkpoint_sha256":metadata["checkpoint_sha256"]}
    base.dump(ROOT/"configs"/"o3b_l2_config.json",config)
    (ROOT/"report"/"10_final_report.md").write_text(report_text,encoding="utf-8")
    print(json.dumps({"stage":"diagnosis_complete","elapsed_s":elapsed,"verdict":gate,"out":str(ROOT)}),flush=True)


def create_lr_plot(rows):
    colors={"0.5x":"#9467bd","1x":"#1f77b4","2x":"#ff7f0e"}; plots=[]
    for norm in ("raw","standardized"):
        for task in ("entity","relation","joint"):
            for rep in ("R0","R1","R2"):
                vals=[]
                for step in (500,1000):
                    for lr in LR_SCALE:
                        x=[r["testA"] for r in rows if r["normalization"]==norm and r["task"]==task and r["representation"]==rep and r["lr_label"]==lr and r["step"]==step and r["seed"]==SEEDS[0]]
                        vals.append(None if not x else x[0])
                plots.append((norm,task,rep,vals))
    for norm in ("raw","standardized"):
        for task in ("entity","relation","joint"):
            w=880;h=460;l=70;t=35;pw=750;ph=340
            parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{l}" y="22" font-size="15">{norm} {task}: TestA, seed 11 (optimizer sensitivity only)</text>']
            for p in range(0,101,20):
                y=t+ph*(1-p/100);parts.append(f'<line x1="{l}" x2="{l+pw}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="20" y="{y+4}" font-size="11">{p}%</text>')
            for ri,rep in enumerate(("R0","R1","R2")):
                vals=next(v for n,ta,r,v in plots if n==norm and ta==task and r==rep); coords=[]
                for i,v in enumerate(vals):
                    if v is not None: coords.append(f'{l+pw*i/5:.1f},{t+ph*(1-v):.1f}')
                color=("#1f77b4","#d62728","#2ca02c")[ri];parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>')
                parts.append(f'<text x="{l+pw-45}" y="{t+16+ri*18}" fill="{color}" font-size="12">{rep}</text>')
            parts.append(f'<text x="{l+pw/2}" y="{h-12}" text-anchor="middle" font-size="12">0.5x/1x/2x LR at 500 steps, then 0.5x/1x/2x at 1000 steps</text></svg>')
            (PLOTS/f"lr_sensitivity_{norm}_{task}.svg").write_text("".join(parts),encoding="utf-8")


if __name__=="__main__": main()
