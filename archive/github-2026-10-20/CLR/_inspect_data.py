from clutrr_data import generate_dataset, RELATIONS
import random, re
for seed in [42, 100, 200]:
    rng = random.Random(seed)
    data = generate_dataset(10, (2,3), (0.6,0.4), rng)
    for t in data:
        for turn in t['turns']:
            prompt = turn['prompt']
            print(f'seed={seed}: {prompt}')
        break
    print()
