import json
with open('H:/CLR/checkpoints/local_clutrr_v2_nolora/results.json') as f:
    d = json.load(f)
elog = d['eval_log'][3]
print('Final eval at step 400:')
ws = elog['with_S']['t2p']
wo = elog['without_S']['t2p']
print(f'  with_S t2p: {ws:.3f}')
print(f'  without_S t2p: {wo:.3f}')
delta = (ws - wo) * 100
print(f'  delta: {delta:+.1f}pp')
