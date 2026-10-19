import json
from pathlib import Path
p=Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bi_final")
def j(n): return json.loads((p/n).read_text(encoding='utf-8'))
def pct(x): return 'n/a' if x is None else f'{100*float(x):.2f}%'
r=j('variant_results.json'); pr=j('probe_results.json'); eq=j('entity_equality.json'); geo=j('entity_geometry.json'); st=j('state_selection_contribution.json'); er=j('error_audit.json'); mar=j('margin_analysis.json')
print('--- variants')
for n,v in r.items(): print(n,'A F1',pct(v['testA']['f1']),'B F1',pct(v['testB']['f1']),'A rec',pct(v['testA']['recall']),'B rec',pct(v['testB']['recall']),'A P1',pct(v['testA']['per_category_accuracy']['P1_same_key']['accuracy']),'B P1',pct(v['testB']['per_category_accuracy']['P1_same_key']['accuracy']),'A P2',pct(v['testA']['per_category_accuracy']['P2_same_entity_diff_relation']['accuracy']),'B P2',pct(v['testB']['per_category_accuracy']['P2_same_entity_diff_relation']['accuracy']),'sameB',pct(v['same_entity_relation']['testB']['accuracy']),'fixedB',pct(v['entity_fixed_relation']['testB']['accuracy']),'A auc',pct(v['testA']['auroc']),'B auc',pct(v['testB']['auroc']),'A pr',pct(v['testA']['pr_auc']),'B pr',pct(v['testB']['pr_auc']))
print('--- input/stage probes')
for n,v in pr.items():
 print(n, 'inputE',pct(v['input']['entity']['aggregate']['testA']['mean']),'inputR',pct(v['input']['relation']['aggregate']['testA']['mean']),'firstE',pct(v['first_projection']['entity']['aggregate']['testA']['mean']),'preE',pct(v['pre_norm_K']['entity']['aggregate']['testA']['mean']),'KE',pct(v['K']['entity']['aggregate']['testA']['mean']),'KR',pct(v['K']['relation']['aggregate']['testA']['mean']))
print('--- equality')
for n,v in eq.items(): print(n,'A',pct(v['testA']['f1']),'B',pct(v['testB']['f1']),'aucB',pct(v['testB']['auroc']),'prB',pct(v['testB']['pr_auc']),'overB',geo[n]['testB']['cosine_overlap'])
print('--- state',st)
print('--- error best', {k:er['testB'].get(k) for k in ['error_counts','error_fractions','margin']})
print('--- margin best',mar[er['best_variant']]['testB'])
print('--- gate',j('gate_evaluation.json'))
