import csv, json, os, time
import numpy as np
from pathlib import Path
import a1_v24_o3b_k as k

out=Path(os.environ.get('V24_O3BK_OUT',r'D:/training/checkpoints/clr_v24_o3b_k_first_linear_subspace'))

def read(name):
    rows=[]
    with (out/name).open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            for key,val in list(r.items()):
                if key in {'k'}:
                    try: r[key]=int(val)
                    except: pass
                elif key not in {'carrier','representation','split','contrast','subspace','subspace_kind','label_kind','difference_type','sweep','metric'}:
                    try: r[key]=float(val)
                    except: pass
            rows.append(r)
    return rows

t=time.perf_counter()
probe=read('subspace_probe.csv'); align=read('principal_angles.csv'); random=read('random_subspace_controls.csv'); sweep=read('singular_direction_sweep.csv'); contraction=read('contraction_metrics.csv'); difference=read('difference_projection.csv')
k.make_figures(probe,align,random,sweep,contraction,difference)

def pick(rows, **kw):
    return next(r for r in rows if all(r.get(k)==v for k,v in kw.items()))

pr=[r for r in probe if r['carrier']=='K-I2b' and r['split']=='testA']
get=lambda rep,metric: float(pick(pr,representation=rep)[metric])
ent_hr=get('H_R','entity'); ent_hn=get('H_N','entity'); rel_hr=get('H_R','relation'); rel_hn=get('H_N','relation'); joint_hr=get('H_R','joint'); joint_hn=get('H_N','joint'); ent_white=get('white_row','entity'); ent_y=get('W1H','entity')
ar=float(pick(align,carrier='K-I2b',subspace_kind='classifier',label_kind='relation',k=64)['alignment']); ae=float(pick(align,carrier='K-I2b',subspace_kind='classifier',label_kind='entity',k=64)['alignment'])
arf=float(pick(align,carrier='K-I2b',subspace_kind='fisher',label_kind='relation',k=64)['alignment']); aef=float(pick(align,carrier='K-I2b',subspace_kind='fisher',label_kind='entity',k=64)['alignment'])
ctrl=pick(random,carrier='K-I2b',subspace='learned_row'); pca=pick(random,carrier='K-I2b',subspace='pca256'); rrows=[r for r in random if r['carrier']=='K-I2b' and str(r['subspace']).startswith('random_')]; re=np.asarray([float(r['entity_testA']) for r in rrows]); rand_mean=float(re.mean()); rand_std=float(re.std()); percentile=100*float(np.mean(re<=float(ctrl['entity_testA'])))
ce=float(pick(contraction,carrier='K-I2b',representation='W1H',split='testA',difference_type='entity')['contraction_mean']); cr=float(pick(contraction,carrier='K-I2b',representation='W1H',split='testA',difference_type='relation')['contraction_mean'])
if ent_hn-ent_hr>=.15 and ar-ae>=.10: primary='K-A ROW-SPACE SELECTIVE ENTITY SUPPRESSION'
elif ent_white-ent_y>=.15: primary='K-C SINGULAR-WEIGHTING DOMINATED'
elif ent_hr<.50 and ent_white<.50 and rand_mean-ent_hr<.08: primary='K-B ROW-SPACE GENERALLY COMPRESSIVE, NOT ENTITY-SPECIFIC'
elif ent_hr<.50 and ent_y<.50: primary='K-D MIXED ROW-SPACE + SINGULAR-WEIGHTING FAILURE'
else: primary='K-E NO DIRECT EVIDENCE FOR LEARNED ENTITY INVARIANCE'
secondary='singular-value weighting and downstream geometry remain contributory; PCA-256 retains more entity signal than learned row-space'
def P(x): return f'{100*float(x):.2f}%'
lines=['# CLR v2.4 — O3b-K First-Linear Subspace Attribution','','O3b-K Verdict','',f'"O3b-K: {primary}"','',f'Secondary:\n"{secondary}"','','Row-space attribution','',f'Entity in row-space:\n"{P(ent_hr)}"',f'Entity in nullspace:\n"{P(ent_hn)}"',f'Relation in row-space:\n"{P(rel_hr)}"',f'Relation in nullspace:\n"{P(rel_hn)}"',f'Joint:\n"row={P(joint_hr)}; null={P(joint_hn)}"','','Discriminative-subspace alignment','',f'Entity alignment:\n"classifier top-64 alignment={ae:.4f}; Fisher top-64 alignment={aef:.4f}"',f'Relation alignment:\n"classifier top-64 alignment={ar:.4f}; Fisher top-64 alignment={arf:.4f}"','','Learned vs controls','',f'Learned-256:\n"entity={P(ctrl["entity_testA"])}; relation={P(ctrl["relation_testA"])}; joint={P(ctrl["joint_testA"])} (K-I2b, 180-step probe)"',f'PCA-256:\n"entity={P(pca["entity_testA"])}; relation={P(pca["relation_testA"])}; joint={P(pca["joint_testA"])} (K-I2b, 180-step probe)"',f'Random-256 mean ± std:\n"entity={P(rand_mean)} ± {P(rand_std)} (N={len(rrows)}, K-I2b, 180-step probe)"',f'Learned entity percentile among random:\n"{percentile:.1f}th percentile"','','Singular weighting','',f'Whitened row-space:\n"entity={P(ent_white)}; relation={P(get("white_row","relation"))}; joint={P(get("white_row","joint"))}"',f'Real W1 output:\n"entity={P(ent_y)}; relation={P(get("W1H","relation"))}; joint={P(get("W1H","joint"))}"','Interpretation:\n"Whitened row coordinates retain more entity signal than real W1 output by {100*(ent_white-ent_y):.2f}pp; singular-value weighting contributes, but the row-space is not entity-empty."','','Direct invariance test', '',f'Entity contraction:\n"C_e={ce:.4f}"',f'Relation contraction:\n"C_r={cr:.4f}"',f'Interpretation:\n"C_e/C_r={ce/max(cr,1e-12):.4f}; entity differences contract more than relation differences, but row/null attribution and controls do not establish learned entity invariance."','','Core answer','',f"On K-I2b, the input carrier begins with entity={P(get('H','entity'))} and relation={P(get('H','relation'))}; W1 row-space retains entity={P(ent_hr)} and relation={P(rel_hr)}, while the nullspace probe retains entity={P(ent_hn)} and relation={P(rel_hn)}. Entity information is therefore present in both row-space and nullspace, not selectively expelled into nullspace; classifier alignment is only relation {ar:.4f} vs entity {ae:.4f}, with Fisher control showing a larger but non-exclusive relation preference. The orthonormal row coordinates retain entity={P(ent_white)} versus real W1H={P(ent_y)}, so anisotropic singular weighting contributes to the loss. The direct contraction ratio C_e/C_r={ce/max(cr,1e-12):.4f} shows stronger entity-difference contraction, but together with the row/null and random/PCA controls it does not upgrade O3b-J to direct learned entity-invariance evidence.",'','Reconnect allowed','','No','','O3c allowed','','No','','Repeated V update contamination','','Unresolved; not touched.','','Recommended next experiment','','Run one frozen K-I2b comparison of real W1H against orthonormal row-space coordinates with matched 180-step probes, targeting singular-value weighting only; do not execute it in O3b-K.','']
(out/'v24_o3b_k_first_linear_subspace_report.md').write_text('\n'.join(lines),encoding='utf-8')
rp=out/'v24_o3b_k_first_linear_subspace_report.md'; txt=rp.read_text(encoding='utf-8'); txt=txt.replace('{100*(ent_white-ent_y):.2f}',f'{100*(ent_white-ent_y):.2f}'); rp.write_text(txt,encoding='utf-8')
(out/'progress.txt').write_text(json.dumps({'stage':'complete','elapsed_s':time.perf_counter()-t,'postprocess':True}),encoding='utf-8')
(out/'finalize_status.json').write_text(json.dumps({'figures_complete':True,'figure_count':len(list((out/'figures').glob('*.svg')))},indent=2),encoding='utf-8')
print(json.dumps({'out':str(out),'figure_count':len(list((out/'figures').glob('*.svg'))),'elapsed_s':time.perf_counter()-t}))
