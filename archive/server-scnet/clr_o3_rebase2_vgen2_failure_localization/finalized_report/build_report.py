"""Assemble source-grounded contract diff and conservative final verdict."""
import ast, csv, hashlib, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def csvout(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
def percent(x): return f'{x*100:.2f}%'

def main():
 old=ROOT/'clr_v24_o3b_m3_end_to_end_reconnect/source_reference/a1_v24_vgen.py'
 three=ROOT/'clr_o3_rebase1_fresh_operational_baseline/components/vgen2/vgen2_rebase_config.json'
 ten=ROOT/'clr_o3_rebase1_vgen2_10k_diag/vgen2_10k_frozen_config.json'
 nc=read(three); gc=read(ten); audit=read(OUT/'prepare/input_audit.json'); paired=read(OUT/'prepare/paired_replay.json'); source_audit=read(OUT/'source_audit/source_split_audit.json')
 fields={
 'input_tensor_source':('MR cache fact_H + LS0 old replay H','MR cache fact_H, certified bitwise equal to J3C H_base_I0'),
 'shape_dtype':('MR [batch,N,2048] -> float32; cached dtype unknown for historic run','[batch,N,2048], cast float32'),
 'normalization':('No normalization in recovered VGen core/decoder; extraction details unknown','No VGen normalization; probe-only train-stat standardization recorded separately'),
 'backbone_layer_token':('Historical cache-builder/run overrides unavailable','See prepare/extraction_policy_frozen.json and carrier_construction.json; R0 is cached direct source carrier, not a fresh backbone replay'),
 'K_Q_V_extraction':('K/Q separately encoded; V core linear+GRU; oracle correct-slot value diagnosis','V only from frozen fact_H; K/Q not inputs to value core or decoder'),
 'entity_relation_value_carrier':('Recovered MRCache fact_H and fact_Y; historical binary cache identity unavailable','Certified relation carrier H_base_I0; semantic entity/relation labels used only for update-pair construction'),
 'target_construction':('Query-selected MR Y; LS0 replay Y; T2/T3 mean CE','Mean per-written-slot fact_Y CE; UPDATE average first/latest CE'),
 'vocabulary_mapping':('64 classes; historical ordered vocabulary artifact unavailable','64 classes; rowwise object-to-class consistency measured in input_audit.json'),
 'padding_masking':('Cap-specific batches; slots 0..N-1, zero unoccupied state','Cap-specific batches, random distinct physical slots; no padded targets'),
 'negative_sampling':('No value CE negatives; counterfactual diagnostics exist separately','No sampled value CE negatives; all 64 logits; different-value same-key UPDATE pairs'),
 'N_distribution':('Uniform cap sampling among 1,2,4 by sample_mr; historic run overrides unknown','Uniform cap sampling 1,2,4 in episode branch'),
 'single_multi_ratio':('50% LS0 N=1 replay plus 50% MR for T2/T3','50% episode branch, 50% same-slot UPDATE branch'),
 'UPDATE_construction':('No same-slot latest-value UPDATE objective in recovered VGen staged training','Same entity/relation, different source facts, object values and labels; all unordered changed-value pairs'),
 'train_val_split':('Cached train/val; historical cache hashes/split lineage unavailable','Zero canonical fact overlap; row joins checked; episode IDs not encoded in NPZ'),
 'random_seed':('Default 42; stage-specific RNG seed=42+sum(ord(name))%10000','42; training RNG seed=2442'),
 'optimizer':('AdamW, stages differ in trainable parameter sets','AdamW'),
 'LR':(0.0003,nc['lr']),
 'batch_size':('MR up to 64 plus old replay 32 for T2/T3',nc['batch_size']),
 'scheduler':('None in recovered source','None'),
 'steps':('Default 3000 per stage; historical exact budget and best step unknown',nc['steps']),
 'initialization':('load_r2_checkpoint: pretrained R2 core; T3 starts from MR trained value decoder','Fresh random core and heads; R2 state unavailable'),
 'frozen_unfrozen':('T1 decoder only, T2 core only, T3 core+decoder; backbone frozen','Core and four value heads jointly trainable; upstream components frozen'),
 'selection_metric':('Validation macro queried-answer accuracy; exact VGen2 source/passing run absent',nc['selection']),
 'validation_randomness':('Fixed physical positions in recovered source','3K varies eval slot permutation by step; 10K fixes permutation (evaluation only)'),
 'historical_identity':('O3 loaders require clr_v24_vgen2/best.pt; recovered staged trainer defaults to clr_v24_vgen/t*_best.pt','Explicit rebase component vgen2_rebase.pt'),
 }
 rows=[{'field':k,'recovered_legacy_source':a,'rebase_3k':b,'status':'DIFFERENT' if a!=b else 'SAME','historical_actual_run_authenticated':False} for k,(a,b) in fields.items()]
 drift={'comparison':'Recovered legacy source contract vs explicitly new rebase baseline; legacy source is NOT certified as the historical passing VGen2 implementation','historical_checkpoint_found':False,'historical_commit_found':False,'rows':rows,
        'rebase_3k_to_10k_config_diff':{k:{'3k':nc.get(k),'10k':gc.get(k)} for k in sorted(set(nc)|set(gc)) if nc.get(k)!=gc.get(k)}}
 dump(OUT/'contract_diff.json',drift); csvout(OUT/'contract_diff.csv',rows)
 # Verify archived passing weights against their source artifact manifest, not just a new digest.
 manifest=read(ROOT/'clr_o3_rebase1_vgen2_10k_diag/artifact_sha256_manifest.json')
 expected={x['path']:x['sha256'] for x in manifest['files']}; verified=[]
 for p in sorted((OUT/'downloads/vgen2_10k').rglob('*.pt')):
  rel=p.relative_to(OUT/'downloads/vgen2_10k').as_posix(); actual=sha(p)
  if expected.get(rel)!=actual: raise RuntimeError('Archived checkpoint manifest mismatch: '+rel)
  verified.append({'path':rel,'sha256':actual,'original_manifest_matches':True})
 dump(OUT/'passing_weight_lineage_verification.json',{'verified':verified,'count':len(verified)})
 prefix=read(OUT/'lineage/3k_prefix_exact_replay.json')
 probes=[]
 for p in sorted(OUT.glob('probe_*/metrics.json')):
  r=read(p); row={'snapshot':r['snapshot'],'stage':r['stage'],'kind':r['kind']}
  row.update({k:v['accuracy'] for k,v in r['metrics'].items()}); probes.append(row)
 csvout(OUT/'probe_table.csv',probes)
 sanity=[]
 for p in sorted(OUT.glob('sanity_*/metrics.json')):
  r=read(p); final=r['curves'][-1]; sanity.append({'size':r['size'],'lr_scale':r['lr_scale'],'train_accuracy':final['train_accuracy_all_slots'],'val_overall':final['val_single']['overall']['answer_accuracy'],'val_update':final['val_update']['latest_value_accuracy_after_update'],'steps':r['steps']})
 if sanity: csvout(OUT/'optimization_sanity_table.csv',sanity)
 # Same hidden width and objective for R0-R4; input dimensions differ by representation.
 localization=[]
 for r in probes:
  if r['kind']=='mlp':
   g=audit[r['snapshot']]['geometry'][r['stage']]
   localization.append({**r,'within_class_cosine':g['within_class_cosine']['mean'],'between_class_cosine':g['between_class_cosine']['mean'],'pairwise_auc':g['same_vs_different_cosine_auc']})
 csvout(OUT/'localization_table.csv',localization)
 three_mlp={r['stage']:r for r in probes if r['snapshot']=='3k' and r['kind']=='mlp'}
 cliffs=[]
 for left,right in [('R0','R1'),('R1','R2'),('R2','R3'),('R3','R4')]:
  cliffs.append({'edge':left+' -> '+right,'single_write_drop_pp':100*(three_mlp[left]['single_write']-three_mlp[right]['single_write']),'update_latest_drop_pp':100*(three_mlp[left]['UPDATE_latest']-three_mlp[right]['UPDATE_latest'])})
 dump(OUT/'probe_cliffs.json',{'edges':cliffs,'interpretation':'Fixed probe readout differences; not proof of irreversible information loss or a specific implementation defect'})
 # Full lineage of the claimed historical passing implementation is still absent.
 # Recipe differences in a candidate legacy source do not prove the unidentified run used them.
 verdict='O3-REBASE2-INCOMPLETE — ROOT CAUSE NOT YET IDENTIFIED'
 verdict_reason='Current 3K shortfall can be localized using paired rebase checkpoints. The exact historical passing VGen2 implementation/checkpoint/dataset/config/commit is not authenticated, so historical regression attribution remains incomplete. Longer training is not a repaired implementation defect; PASS-A is not claimed. Candidate legacy-source drift alone cannot prove FAIL-CONTRACT for an unidentified passing run.'
 dump(OUT/'final_verdict.json',{'verdict':verdict,'reason':verdict_reason,'historical_attribution_complete':False,'current_rebase_diagnosis_complete':True,'allocator_allowed':False,'router_memory_reconnect_allowed':False,'integrated_o3_allowed':False,'testA_testB_opened_this_round':False})
 (OUT/'FINAL_VERDICT.txt').write_text(verdict+'\n',encoding='utf-8')
 hashes={str(p.relative_to(ROOT)):sha(p) for p in [old,three,ten,OUT/'diagnose.py',OUT/'run_diagnostics.py',OUT/'audit_source_split.py',ROOT/'tools/remote_weights.py',ROOT/'tools/train_and_pull.py',ROOT/'AGENTS.md',ROOT/'SERVER_CONNECTION.md']}
 dump(OUT/'source_config_hashes.json',{'git_sha':None,'reason':'H:/LoopLM is not a Git repository; no historical commit recovered. Content hashes replace provenance guesses.','sha256':hashes,'remote_dataset_checkpoint_hashes':'prepare/input_audit.json'})
 lines=['# O3-REBASE2 VGen2 failure localization','',f'**{verdict}**','',verdict_reason,'',
 '## Baseline correction and measured attribution','',
 'The pasted O3-REBASE1-F freeze is superseded by an archived 10K standalone passing run. All 18 archived checkpoints were re-downloaded from the current server and checked against their original SHA-256 manifest. This diagnosis does not authorize downstream work.','',
 f"The original failing 3K checkpoint and the 10K run's step-3000 snapshot have **all core/head tensors exactly equal={prefix['all_exact']}**, max absolute difference 0. This controls initialization and the first 3000 optimization updates: the later passing checkpoint is obtained by further optimization in the same training trajectory, not an input/label substitution.",'',
 '| Checkpoint | Step | Single-write overall | N=4 | UPDATE latest |','|---|---:|---:|---:|---:|']
 for name in ('3k','10k'):
  r=audit[name]; lines.append(f"| {name} | {audit['artifacts'][name]['step']} | {percent(r['single']['overall']['answer_accuracy'])} | {percent(r['single']['4']['answer_accuracy'])} | {percent(r['update']['latest_value_accuracy_after_update'])} |")
 lines+=['',f"3K checkpoint SHA-256: `{audit['artifacts']['3k']['sha256']}`.",f"10K selected checkpoint SHA-256: `{audit['artifacts']['10k']['sha256']}`.",f"3K config SHA-256: `{sha(three)}`; 10K config SHA-256: `{sha(ten)}`.",f"Train MR dataset SHA-256: `{audit['source_hashes']['/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/train_data.npz']}`.",f"VAL MR dataset SHA-256: `{audit['source_hashes']['/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache/val_data.npz']}`."]
 lines+=['','Identical frozen validation source rows and labels are replayed through both checkpoints. Same deterministic physical slot assignments are used for the paired per-example representation capture; the standalone metric table also uses a fixed validation permutation. Small differences from archived 3K metrics can therefore reflect only a changed evaluation permutation, not a changed dataset.','',
 '## Lineage / contract audit','',
 'A true historical passing VGen2 cannot currently be recovered. O3 source loaders name `clr_v24_vgen2/best.pt`, while recovered `a1_v24_vgen.py` writes staged `clr_v24_vgen/t*_best.pt`. The latter is a candidate related source, not authenticated passing lineage. The missing R2 initialization and VGen2 artifact were already documented in M4; this round checks the active project again.','',
 'The recovered legacy source uses pretrained R2 initialization and LS0 single-relation replay. Rebase explicitly uses fresh initialization and episode/UPDATE supervision. Historical queried-answer macro evaluation also differs from current all-written-values and latest-overwrite gates. Thus a numeric historical score must not be treated as a like-for-like 3K rebase target. Exact historical cache identities, label order, run overrides and commit remain unknown.','',
 '`contract_diff.json` and `.csv` cover input source, shapes/dtypes, normalization, extraction, carriers, labels/vocabulary, masks, negatives, N, write/update mixture, splits, seed, optimizer/LR/batch/scheduler/steps, initialization, frozen components, and selection. Unknown historical evidence is marked explicitly.','',
 'Extraction qualification: the copied J3C policy describes the **fresh entity-mention branch** and composite `H_fresh=H_base_I0+2*H_mention_fresh`. VGen2 consumes **H_base_I0 / MR fact_H**, not that composite. The MR cache manifest only certifies frozen Ouro, float32, 2048 dimensions and cache schema; the original fact_H layer/token selection and builder commit remain unrecorded. Do not silently assign the entity-mention pooling policy to the value source. `lineage/mr_manifest.json` and scoped source hashes retain this distinction.','',
 f"Current train/VAL canonical fact overlap: {audit['canonical_fact_overlap']}. Value mapping conflicts: {len(audit['value_mapping_conflicts'])}. R0/R1 bitwise equality: {audit['3k']['R0_R1_exact']}. Rowwise MR/certified carrier equality and label/sidecar ID joins pass. Canonical IDs include split prefixes, so zero ID overlap alone does not prove source separation. The supplemental source-text audit finds **{source_audit['cross_split']['unique_shared_exact_fact_texts']} shared exact fact texts / {source_audit['cross_split']['validation_fact_rows']} VAL fact rows**, and {source_audit['cross_split']['shared_semantic_entity_relation_value_triples']} shared semantic triples. These small duplicate-source overlaps are disclosed; the frozen VAL set is not altered for primary metrics.",'',
 f"Excluding exact train-text duplicates from reporting only, raw linear single-write={percent(source_audit['probe_3k_R1_linear']['single_unseen_exact_text']['accuracy'])}, UPDATE={percent(source_audit['probe_3k_R1_linear']['update_unseen_latest_exact_text']['accuracy'])}. Train/VAL query-target offset mismatches and metadata fact-class mismatches are all zero. See `source_audit/source_split_audit.json` for array shapes/dtypes and denominators. Historical contamination cannot be ruled out for missing historical caches. The backbone was not re-extracted.",'',
 '## Raw-carrier supervised diagnostic probes','',
 '| Probe | Value identity incl. UPDATE contexts | N=1 | N=2 | N=4 | Single-write | UPDATE latest |','|---|---:|---:|---:|---:|---:|---:|']
 for r in probes:
  if r['snapshot']=='3k' and r['stage']=='R1': lines.append('| '+r['kind']+' | '+' | '.join(percent(r[k]) for k in ['value_identity','N1','N2','N4','single_write','UPDATE_latest'])+' |')
 lines+=['','All fits use train only; VAL is evaluated once at the fixed final step. Linear/MLP use train-only feature means/scales, AdamW LR 0.003, WD 0.01, seed 42, batch 256 (half single-write, half UPDATE contexts), 2000 steps. MLP hidden width is 64. Centroids fit cosine means on train only. UPDATE raw input is the latest incoming fact; raw probes do not emulate a recurrent overwrite cell. A failed finite-capacity probe is not proof of information-theoretic insufficiency.','',
 '## R0–R4 information preservation','',
 '| Snapshot | Stage | Single-write probe | UPDATE probe | Within-value cosine | Between-value cosine | Pair AUC |','|---|---|---:|---:|---:|---:|---:|']
 for r in localization: lines.append(f"| {r['snapshot']} | {r['stage']} | {percent(r['single_write'])} | {percent(r['UPDATE_latest'])} | {r['within_class_cosine']:.4f} | {r['between_class_cosine']:.4f} | {r['pairwise_auc']:.4f} |")
 worst_single=max(cliffs,key=lambda x:x['single_write_drop_pp']); worst_update=max(cliffs,key=lambda x:x['update_latest_drop_pp'])
 lines+=['',f"Largest measured MLP readout drop: single-write **{worst_single['edge']} ({worst_single['single_write_drop_pp']:.2f}pp)**; UPDATE **{worst_update['edge']} ({worst_update['update_latest_drop_pp']:.2f}pp)**. Raw linear exceeds 95%, while the same-budget raw MLP is lower; therefore finite-probe optimization/generalization is a real confound. These stage differences localize degraded readout but do not prove irreversible information loss. The 10K R3 probe recovers the operational region without altering source inputs or architecture."]
 lines+=['','R0 = certified cached source state; R1 = float32 core input; R2 = content_read linear output; R3 = GRU value state (including same-slot UPDATE history); R4 = selected slot decoder logits. The same examples are saved in `prepare/3k_val.npz` and `10k_val.npz`; train representations remain SHA-manifested on the server. Identical two-layer MLP architecture family (hidden width 64, 64 outputs, same optimizer/budget) is used at every stage; input dimensions differ (2048 vs 64), so parameter totals are reported and are not falsely claimed equal. Geometry uses frozen validation representations and fixed sampled non-self pairs; these cosine statistics are descriptive, not an optimal separability bound.','',
 '## Paired replay and failure taxonomy','',
 f"3K failing vs 10K passing: source input equal={paired['raw_input_equal']}; labels equal={paired['labels_equal']}; passing-correct/failing-wrong={paired['passing_correct_failing_wrong']}; both wrong={paired['both_wrong']}; failing-correct/passing-wrong={paired['failing_correct_passing_wrong']}.",
 f"Maximum projection difference={paired['projection_max_abs_diff']:.6f}; maximum latent difference={paired['latent_max_abs_diff']:.6f}. Changed learned transformation, not changed input/label, explains these paired differences. This comparison is not substituted for unavailable historical checkpoint replay.",'',
 'Full entity/relation/value and overwrite-versus-single-write taxonomy is in `prepare/paired_replay.json`; every paired prediction is in `paired_replay.csv`. The largest error classes should be read as frequencies with denominators, not presumed causal defects.','',
 '## Optimization sanity','']
 if sanity:
  lines+=['A raw supervised probe clears the required information gate (linear can establish sufficiency even when the MLP generalizes less well), so fixed 2000-step overfit diagnostics run from the failing 3K checkpoint, using 256 or 1024 unique train facts and all four physical heads. This is a single-write optimization diagnostic; UPDATE is evaluated on frozen VAL, not fit in these runs. LR controls at 0.25x / 1x / 4x are diagnostic contrasts, never selected for production.','',
          '| Unique train facts | LR multiplier | Train accuracy across slots | VAL overall | VAL UPDATE |','|---:|---:|---:|---:|---:|']
  for r in sanity: lines.append(f"| {r['size']} | {r['lr_scale']} | {percent(r['train_accuracy'])} | {percent(r['val_overall'])} | {percent(r['val_update'])} |")
  lines+=['','Per-run curves, gradient norms before clipping, class histograms, prediction entropy and final confusion matrices are saved with each checkpoint. No VGen production architecture, Candidate B, Rule A, or upstream state is changed.']
 else: lines+=['NOT RUN: the raw MLP did not clear the predeclared diagnostic gate. Do not continue VGen optimization based on negative finite-probe evidence alone.']
 lines+=['','## Verdict and next permitted step','',
 'The current rebase carrier and failing learned transformation can be diagnosed from the measurements above; the archived longer run already demonstrates that current inputs support the operational gate. The 3K experiment is a fresh-baseline training-budget failure, not evidence that the carrier cannot encode values. However, no implementation defect has been localized and repaired, and the true historical passing contract has not been authenticated. Therefore strict PASS-A/FAIL-REP/FAIL-CONTRACT requirements are not claimed; the overall historical-root-cause task remains INCOMPLETE.','',
 'Permitted: recover the exact historical `clr_v24_vgen2/best.pt`, its training source/config and R2 initialization/cache artifacts, then replay the same train/VAL rows under that authenticated contract; or explicitly accept the SHA-pinned 10K fresh rebase as the new standalone baseline in a later instruction. Prohibited: allocator, router/memory reconnect, integrated O3 evaluation, TestA/TestB tuning, changing Candidate B/Rule A, or redesigning the architecture to mask attribution.','',
 '## Reproducibility and downloads','',
 'Git SHA: **N/A**, the local workspace is not a Git repository and no historical commit was recovered. Do not invent a commit ID. Dataset, config, source and checkpoint content hashes are in `prepare/input_audit.json`, `source_config_hashes.json`, the original checkpoint manifest verification, and each run manifest. Current server: `ssh.zzai.scnet.cn:11810`, root; runtime torch 2.9.0 / BW GPU with the pinned DTK library path. Password is never included in experiment artifacts.','',
 'Each training task runs through `tools/train_and_pull.py`: close its SSH connection, reconnect, download checkpoints/config/metrics, validate server SHA-256, atomically promote downloads, and save `download_verification.json` before the next training task. Existing differing local weights are preserved. Project `AGENTS.md` records this policy for future training turns.','',
 'Resume existing artifacts: run local `run_diagnostics.py`; completed task markers allow artifact recovery without silent retraining. For a fresh reproduction, use a separate experiment output root in the scripts, preserve source MR/J3C and checkpoint inputs by SHA-256, run `prepare`, diagnostic tasks, and `audit_source_split.py`, then mirror results. Do not overwrite this frozen run. `build_report.py` verifies lineage and regenerates the report.','']
 report=OUT/'v24_o3_rebase2_vgen2_failure_localization_report.md'; report.write_text('\n'.join(lines),encoding='utf-8')
 for p in sorted(OUT.glob('probe_*/metrics.json')):
  r=read(p); dump(p.parent/'execution_protocol.json',{'source_sha256':sha(OUT/'diagnose.py'),'source_revision_note':'Only sanity admission changed from MLP-only to either supervised raw probe; probe implementation unchanged throughout this round','input_manifest_sha256':sha(OUT/'prepare/manifest.json'),'optimizer':None if r['kind']=='centroid' else {'name':'AdamW','lr':.003,'weight_decay':.01,'gradient_clip':10},'seed':42,'steps':r['steps'],'batch_size':None if r['kind']=='centroid' else 256,'training_mixture':'50% single-write facts, 50% UPDATE latest contexts' if r['kind']!='centroid' else 'all train contexts for class centroids','model_selection':'fixed final step; no VAL selection','hidden_width':64 if r['kind']=='mlp' else None,'normalization':r['preprocessing'],'checkpoint_sha256':sha(p.parent/'probe.pt')})
 for p in sorted(OUT.glob('sanity_*/metrics.json')):
  r=read(p); dump(p.parent/'execution_protocol.json',{'source_sha256':sha(OUT/'diagnose.py'),'initial_checkpoint_sha256':audit['artifacts']['3k']['sha256'],'information_gate':'raw linear single>=95% and UPDATE>=95%; negative MLP does not negate linear positive evidence','optimizer':{'name':'AdamW','lr':.0003*r['lr_scale'],'weight_decay':.01,'gradient_clip':1},'seed':42,'steps':2000,'batch_size':64,'unique_train_fact_count':r['size'],'objective':'single-write CE; random physical slot each step','model_selection':'fixed final step; no VAL selection','checkpoint_sha256':sha(p.parent/'sanity.pt')})
 files=[{'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'artifact_sha256_manifest.json'} and p.suffix not in {'.pyc','.partial'}]
 dump(OUT/'artifact_sha256_manifest.json',{'files':files,'git_sha':None,'manifest_excludes_itself':True})
 print(verdict); print(report)

if __name__=='__main__': main()
