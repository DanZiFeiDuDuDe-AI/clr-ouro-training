from clutrr_data import generate_dataset, RELATIONS
import random

rng = random.Random(42)
data = generate_dataset(10, (2,3), (0.6,0.4), rng)
for t in data[:2]:
    print('=== Trajectory ===')
    n_turns = len(t['turns'])
    ans = t['turns'][-1]['answer']
    print(f'  Total turns: {n_turns}')
    for i, turn in enumerate(t['turns']):
        print(f'  Turn {i}: prompt="{turn["prompt"][:80]}..." ans={turn["answer"]}')
    # Analyze: can we compute reasoning depth / composed relations?
    relations_mentioned = []
    for turn in t['turns']:
        for r in RELATIONS:
            if r.lower() in turn['prompt'].lower():
                relations_mentioned.append(r)
    print(f'  Relations in prompt: {relations_mentioned}')
    print()
