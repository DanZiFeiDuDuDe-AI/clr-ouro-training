import json
with open('H:/CLR/02_results/phase_d_results.json') as f:
    d = json.load(f)['DS']

for ds in sorted(d.keys(), key=int):
    r = d[ds]
    bd = r.get('best_delta_t2p', 0) * 100
    fd = r.get('final_delta_t2p', 0) * 100
    ws = r.get('final_with_S_t2p', 0) * 100
    wo = r.get('final_without_S_t2p', 0) * 100
    cl = r.get('bayes_ceiling_t2p', 0) * 100
    elog = r.get('eval_log', [])
    deltas = ' | '.join(f"s{e['step']}={e['delta']['t2p']*100:+.1f}pp" for e in elog if 'delta' in e)
    norms = ' | '.join(f"s{e['step']}={e.get('s_norm', 0):.3f}" for e in elog if 's_norm' in e)
    print(f'DS={ds:>4s} | Best={bd:+7.2f}pp | Final={fd:+7.2f}pp | ceil={cl:5.1f}% | w/S={ws:5.1f}% | w/o S={wo:5.1f}%')
    print(f'     deltas: {deltas}')
    if norms:
        print(f'     |S|:     {norms}')
