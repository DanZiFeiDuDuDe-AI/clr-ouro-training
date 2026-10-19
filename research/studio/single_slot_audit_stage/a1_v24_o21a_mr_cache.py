# -*- coding: utf-8 -*-
"""CLR-v2.4-O2.1a-MR frozen Ouro cache builder.

This is a new multi-relation cache.  It deliberately does not relabel the old
single-relation LS2 cache.  Facts and query strings are encoded by the frozen
Ouro model and only the resulting 2048-d hidden vectors are persisted.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/training")
sys.path.insert(0, str(ROOT))
os.environ["OURO_MODEL_PATH"] = os.environ.get("OURO_MODEL_PATH", "D:/training/Ouro-1.4B")

from a1_probe_common import CITIES, CITY2ID, make_person_pool  # noqa: E402
from a1_ls2_cache import extract_h_batch  # noqa: E402
from clr_v2 import load_ouro  # noqa: E402


OUT = Path(os.environ.get(
    "V24_O21A_MR_CACHE",
    r"D:/training/checkpoints/clr_v24_o21a_mr/cache",
))
SEED = int(os.environ.get("V24_O21A_MR_SEED", "42"))
SMOKE = os.environ.get("A1_V24_O21A_MR_SMOKE", "0") == "1" or "--smoke" in sys.argv
DM = 2048
CAPS = (1, 2, 4)
SPLITS = ("train", "val", "testA", "testB")
FORMAL = {"train": 5000, "val": 1000, "testA": 1000, "testB": 1000}
SMOKE_SIZES = {"train": 256, "val": 128, "testA": 128, "testB": 128}

RELATIONS = (
    "current_city",
    "birth_city",
    "favorite_city",
    "work_city",
)
RELATION_ID = {x: i for i, x in enumerate(RELATIONS)}


def fact_text(person, relation, city, lexical=False, variant=0):
    if lexical:
        templates = {
            "current_city": f"{person}的当前城市是{city}。",
            "birth_city": f"{person}的出生城市是{city}。",
            "favorite_city": f"{person}的偏好城市是{city}。",
            "work_city": f"{person}的工作城市是{city}。",
        }
    elif variant % 2 == 0:
        templates = {
            "current_city": f"{person}目前住在{city}。",
            "birth_city": f"{person}出生在{city}。",
            "favorite_city": f"{person}最喜欢的城市是{city}。",
            "work_city": f"{person}在{city}工作。",
        }
    else:
        templates = {
            "current_city": f"{person}现在居住的城市是{city}。",
            "birth_city": f"{person}的出生地是{city}。",
            "favorite_city": f"{person}偏爱的城市为{city}。",
            "work_city": f"{person}的工作地点在{city}。",
        }
    return templates[relation]


def query_text(person, relation, variant=0, lexical=False):
    if lexical:
        templates = {
            "current_city": f"{person}的当前城市是哪座？",
            "birth_city": f"{person}的出生城市是哪座？",
            "favorite_city": f"{person}的偏好城市是哪座？",
            "work_city": f"{person}的工作城市是哪座？",
        }
    elif variant % 2 == 0:
        templates = {
            "current_city": f"{person}目前住在哪座城市？",
            "birth_city": f"{person}出生在哪座城市？",
            "favorite_city": f"{person}最喜欢哪座城市？",
            "work_city": f"{person}在哪座城市工作？",
        }
    else:
        templates = {
            "current_city": f"{person}现在居住的城市是哪里？",
            "birth_city": f"{person}的出生城市是哪里？",
            "favorite_city": f"{person}偏好的城市是哪一座？",
            "work_city": f"{person}的工作地点在哪座城市？",
        }
    return templates[relation]


def _caps(n, seed):
    out = [CAPS[i % len(CAPS)] for i in range(n)]
    random.Random(seed).shuffle(out)
    return out


def _choose_city(rg, used, person, relation, train_bindings, forbid_train):
    choices = list(CITIES)
    rg.shuffle(choices)
    for city in choices:
        if city in used:
            continue
        if forbid_train and city in train_bindings.get((person, relation), set()):
            continue
        return city
    raise RuntimeError("unable to choose an unseen city binding")


def _episode(split, i, cap, rg, persons_pool, train_bindings, forbid_train,
             train=False):
    # The stress modes are part of the supervised distribution so the model
    # sees same-entity and same-relation collisions during training.
    if cap == 1:
        mode = "single"
    elif cap == 4:
        mode = ("mixed", "same_entity", "same_relation", "same_value",
                "lexical_stress")[i % 5]
    else:
        mode = ("mixed", "same_entity", "same_relation")[i % 3]

    if mode in ("same_entity", "lexical_stress"):
        person = rg.choice(persons_pool)
        rels = list(RELATIONS)[:cap]
        if cap < 4:
            rels = rg.sample(list(RELATIONS), cap)
        people = [person] * cap
    elif mode == "same_relation":
        rel = rg.choice(RELATIONS)
        rels = [rel] * cap
        people = rg.sample(list(persons_pool), cap)
    else:
        people = rg.sample(list(persons_pool), cap)
        rels = rg.sample(list(RELATIONS), cap) if cap < 4 else list(RELATIONS)
        rg.shuffle(rels)

    cities = []
    for p, rel in zip(people, rels):
        if mode == "same_value":
            # This is an explicit diagnostic: equal values must not make the
            # router's retrieval metric trivial.
            city = CITIES[(i * 7 + 11) % len(CITIES)]
        else:
            city = _choose_city(rg, set(cities), p, rel, train_bindings, forbid_train)
        cities.append(city)

    qi = rg.randrange(cap)
    lexical = mode == "lexical_stress"
    variant = 100000 + i if split == "testB" else (10000 + i if split == "testA" else i)
    fact_texts = [fact_text(p, r, c, lexical=lexical, variant=variant + j)
                  for j, (p, r, c) in enumerate(zip(people, rels, cities))]
    fact_query_texts = [query_text(p, r, variant=variant + j, lexical=lexical)
                        for j, (p, r) in enumerate(zip(people, rels))]
    qtext = fact_query_texts[qi]
    assert len(set(zip(people, rels))) == cap
    assert len(set(rels)) == cap if mode in ("same_entity", "lexical_stress") else True
    assert qtext and cities[qi] not in qtext
    assert people[qi] in people and rels[qi] in rels
    return {
        "split": split,
        "episode_id": f"{split}-{i}",
        "capacity": cap,
        "episode_type": mode,
        "query_index": qi,
        "persons": people,
        "relations": rels,
        "relation_ids": [RELATION_ID[x] for x in rels],
        "cities": cities,
        "fact_classes": [int(CITY2ID[c]) for c in cities],
        "fact_texts": fact_texts,
        "fact_query_texts": fact_query_texts,
        "query_person": people[qi],
        "query_relation": rels[qi],
        "query_relation_id": RELATION_ID[rels[qi]],
        "query_text": qtext,
        "target_city": cities[qi],
        "target_class": int(CITY2ID[cities[qi]]),
    }


def build_meta(seed=SEED):
    sizes = SMOKE_SIZES if SMOKE else FORMAL
    n_train = sizes["train"]
    p_train = make_person_pool(max(600, min(1200, n_train // 10)), seed=seed + 11)
    p_testb = make_person_pool(max(600, min(1200, sizes["testB"] // 2)),
                               seed=seed + 505, exclude=p_train)
    train_bindings = {}
    episodes = {}
    rg = random.Random(seed + 101)
    for split_idx, split in enumerate(SPLITS):
        pool = p_testb if split == "testB" else p_train
        forbid = split in ("val", "testA")
        rows = []
        for i, cap in enumerate(_caps(sizes[split], seed + split_idx * 1000)):
            e = _episode(split, i, cap, random.Random(seed + split_idx * 1000003 + i * 101),
                         pool, train_bindings, forbid, train=(split == "train"))
            rows.append(e)
            if split == "train":
                for p, r, c in zip(e["persons"], e["relations"], e["cities"]):
                    train_bindings.setdefault((p, r), set()).add(c)
        episodes[split] = rows
    return episodes, p_train, p_testb


def _extract_unique(model, tok, texts, batch_size=8):
    import torch
    out = {}
    t0 = time.time()
    with torch.no_grad():
        for st in range(0, len(texts), batch_size):
            chunk = texts[st:st + batch_size]
            hs = extract_h_batch(model, tok, chunk)
            for text, h in zip(chunk, hs):
                out[text] = h.numpy().astype(np.float32)
            done = st + len(chunk)
            if done == len(texts) or done % 256 < len(chunk):
                print(f"[mr-cache] hidden {done}/{len(texts)}", flush=True)
    return out, time.time() - t0


def build_cache(force=False):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    sizes = SMOKE_SIZES if SMOKE else FORMAL
    if manifest_path.exists() and not force:
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mf.get("sizes") == sizes and mf.get("cache_version") == "O2.1a-MR-v1":
            print("[mr-cache] existing cache matches", flush=True)
            return mf

    episodes, p_train, p_testb = build_meta(SEED)
    for split in SPLITS:
        with (OUT / f"{split}_meta.jsonl").open("w", encoding="utf-8") as f:
            for e in episodes[split]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    fact_texts = sorted({t for es in episodes.values() for e in es for t in e["fact_texts"]})
    query_texts = sorted({t for es in episodes.values() for e in es for t in e["fact_query_texts"]})
    print(f"[mr-cache] facts={len(fact_texts)} queries={len(query_texts)}", flush=True)
    print("[mr-cache] loading frozen Ouro ...", flush=True)
    model, tok = load_ouro()
    model.eval()
    qbs = int(os.environ.get("V24_O21A_MR_OURO_BATCH", "16"))
    fact_h, fact_sec = _extract_unique(model, tok, fact_texts, qbs)
    query_h, query_sec = _extract_unique(model, tok, query_texts, qbs)
    del model, tok

    train_persons = sorted({p for e in episodes["train"] for p in e["persons"]})
    person_id = {p: i for i, p in enumerate(train_persons)}
    total_facts = 0
    for split in SPLITS:
        es = episodes[split]
        n = len(es)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([e["capacity"] for e in es])
        total = int(offsets[-1]); total_facts += total
        fh = np.zeros((total, DM), dtype=np.float32)
        fqh = np.zeros((total, DM), dtype=np.float32)
        fy = np.zeros(total, dtype=np.int64)
        fr = np.zeros(total, dtype=np.int64)
        fe = np.full(total, -1, dtype=np.int64)
        qh = np.zeros((n, DM), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        ns = np.zeros(n, dtype=np.int64)
        qi = np.zeros(n, dtype=np.int64)
        qr = np.zeros(n, dtype=np.int64)
        qe = np.full(n, -1, dtype=np.int64)
        for i, e in enumerate(es):
            st = int(offsets[i]); ns[i] = e["capacity"]; qi[i] = e["query_index"]
            y[i] = e["target_class"]; qr[i] = e["query_relation_id"]
            qe[i] = person_id.get(e["query_person"], -1)
            qh[i] = query_h[e["query_text"]]
            for j, (p, r, ft, qt, vc) in enumerate(zip(
                    e["persons"], e["relations"], e["fact_texts"],
                    e["fact_query_texts"], e["fact_classes"])):
                k = st + j; fh[k] = fact_h[ft]; fqh[k] = query_h[qt]
                fy[k] = vc; fr[k] = RELATION_ID[r]; fe[k] = person_id.get(p, -1)
        np.savez_compressed(OUT / f"{split}_data.npz", fact_H=fh, fact_query_H=fqh,
                            fact_offsets=offsets, query_H=qh, Y=y, N=ns,
                            query_idx=qi, fact_Y=fy, fact_relation=fr,
                            fact_entity=fe, query_relation=qr, query_entity=qe)

    (OUT / "train_persons.json").write_text(json.dumps(train_persons, ensure_ascii=False), encoding="utf-8")
    (OUT / "relations.json").write_text(json.dumps(list(RELATIONS), ensure_ascii=False, indent=2), encoding="utf-8")
    mf = {
        "cache_version": "O2.1a-MR-v1", "seed": SEED, "smoke": SMOKE,
        "sizes": sizes, "capacities": list(CAPS), "relations": list(RELATIONS),
        "n_relations": len(RELATIONS), "n_classes": len(CITIES),
        "chance": 1.0 / len(CITIES), "hidden_dtype": "float32",
        "hidden_shape": [DM], "total_facts": total_facts,
        "unique_fact_texts": len(fact_texts), "unique_query_texts": len(query_texts),
        "train_persons": len(p_train), "testb_persons": len(p_testb),
        "fact_hidden_wall_sec": round(fact_sec, 2),
        "query_hidden_wall_sec": round(query_sec, 2),
        "wall_sec": round(time.time() - t0, 2),
        "source": "new multi-relation text schema encoded by frozen Ouro",
        "ouro_learnable": False,
        "protocol": "four relation families; same-entity, same-relation, same-value, lexical stress; query contains person+relation only",
    }
    manifest_path.write_text(json.dumps(mf, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[mr-cache] done", json.dumps(mf, ensure_ascii=False), flush=True)
    return mf


if __name__ == "__main__":
    build_cache(force=("--force" in sys.argv) or SMOKE)
