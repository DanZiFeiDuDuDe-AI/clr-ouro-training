import json
from pathlib import Path

p = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bh_final")
def j(n): return json.loads((p/n).read_text(encoding="utf-8"))
def acc(d, split="testA"):
    try: return d["entity_linear"]["aggregate"][split]["mean"]
    except Exception: return None
for n in ["keyencoder_internal_probes.json","query_write_comparison.json"]:
    d=j(n); print("---", n)
    if n.startswith("key"):
        for k,v in d.items(): print(k, "E", acc(v), "R", v["relation_linear"]["aggregate"]["testA"]["mean"], "B-E", v["entity_linear"]["aggregate"]["testB"]["mean"])
    else:
        for k,v in d.items(): print(k, "E", v["entity"]["aggregate"]["testA"]["mean"], "R", v["relation"]["aggregate"]["testA"]["mean"], "B-E", v["entity"]["aggregate"]["testB"]["mean"], "B-R", v["relation"]["aggregate"]["testB"]["mean"])
t=j("token_span_probe_results.json")
print("--- token fact")
for k,v in t.get("fact",{}).items(): print(k, "E", v["entity"]["aggregate"]["testA"]["mean"], "R", v["relation"]["aggregate"]["testA"]["mean"])
print("--- token query")
for k,v in t.get("query",{}).items(): print(k, "E", v["entity"]["aggregate"]["testA"]["mean"], "R", v["relation"]["aggregate"]["testA"]["mean"])
print("--- equality")
for rep,v in j("entity_equality.json").items(): print(rep, "A", v["testA"]["f1"], "B", v["testB"]["f1"], "B auc", v["testB"]["auroc"], "B pr", v["testB"]["pr_auc"])
print("--- geometry")
for rep,v in j("entity_geometry.json").items(): print(rep, "A overlap", v["testA"]["cosine_overlap"], "B overlap", v["testB"]["cosine_overlap"], "B same", v["testB"]["same_entity"]["mean"], "B diff", v["testB"]["different_entity"]["mean"])
print("--- recency", j("recency_bias.json"))
print("--- difficulty", j("difficulty_breakdown.json"))
