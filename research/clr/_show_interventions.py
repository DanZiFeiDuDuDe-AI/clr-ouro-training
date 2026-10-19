import json, os

ckpts = ['local_clutrr_v2_nolora', 'seed_1103', 'seed_130', 'seed_520']
for name in ckpts:
    ipath = 'H:/CLR/checkpoints/' + name + '/interventions.json'
    if os.path.exists(ipath):
        with open(ipath) as f:
            d = json.load(f)
        f_t2p = d['full']['t2p']
        r_t2p = d['reset']['t2p']
        z_t2p = d['zeroed']['t2p']
        ma = d['memoryless_argmax']
        ma_a = ma['acc_on_argmax_wrong']
        ma_n = ma['n_argmax_wrong']
        ma_t = ma['n_t2p']
        ma_aa = ma['argmax_acc']
        ma_ag = ma['agreement']
        print(f'{name:>35s}: full={f_t2p:.3f} reset={r_t2p:.3f} zeroed={z_t2p:.3f} | '
              f'acc_on_wrong={ma_a:.3f} (n={ma_n}/{ma_t}) '
              f'argmax_acc={ma_aa:.3f} agree={ma_ag:.3f}')
    else:
        print(f'{name}: no interventions.json')
