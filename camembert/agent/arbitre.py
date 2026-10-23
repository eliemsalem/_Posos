#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitre : un LLM local (Qwen via Ollama) tranche quand l'ensemble CamemBERT hésite
entre deux intentions.

On prend les questions de la validation croisée (réponses connues) où l'écart entre les
deux premières probabilités est faible, et on compare :
  - le choix de l'ensemble (1er choix),
  - le choix de Qwen entre les deux candidates, à partir des noms et descriptions des
    intentions (agent/intentions.json) et de quelques exemples de chacune.

    ~/anaconda3/envs/hf-agents/bin/python agent/arbitre.py oof [--n 200] [--marge 0.15]
    ~/anaconda3/envs/hf-agents/bin/python agent/arbitre.py test probs.npy --out sub.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import ollama

PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJET))
import glob                           # noqa: E402
import pandas as pd                   # noqa: E402

# (pas d'import de train.py / ensemble.py : l'env hf-agents n'a pas `datasets`)


def load_training_frame():
    x = pd.read_csv(PROJET / "data" / "input_train.csv")
    y = pd.read_csv(PROJET / "data" / "output_train.csv")
    df = x.merge(y, on="ID")
    df["question"] = df["question"].fillna("").astype(str)
    return df


def load_oof(pattern):
    """Probabilités out-of-fold d'un groupe de plis (cf. ensemble.load_oof)."""
    dirs = sorted(Path(d) for d in glob.glob(pattern))
    ids = np.concatenate([np.load(d / "val_ids.npy") for d in dirs])
    probs = np.concatenate([np.load(d / "val_probs.npy") for d in dirs])
    return pd.DataFrame(probs, index=ids).sort_index(), len(dirs)

NOMS = json.loads((PROJET / "agent" / "intentions.json").read_text())


def fiche(i, df):
    ex = df[df["intention"] == i]["question"]
    ex = ex[ex.str.len().between(20, 120)].sample(3, random_state=0)
    return (f"Intention {i} — {NOMS[str(i)]['nom']} : {NOMS[str(i)]['description']}\n"
            + "\n".join(f"  ex. : {q}" for q in ex))


def arbitrer(question, a, b, df):
    """Demande à Qwen de choisir entre les intentions a et b (a = 1er choix de l'ensemble)."""
    prompt = f"""Des questions sur les médicaments sont classées par intention (le type
d'information attendue). Pour la question ci-dessous, choisis la bonne intention entre
les deux candidates.

QUESTION : « {question[:400]} »

{fiche(a, df)}

{fiche(b, df)}

Réponds en JSON : {{"raison": "une phrase", "intention": {a} ou {b}}}"""
    r = ollama.chat(model="qwen2.5-coder:14b", format="json",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_ctx": 4096})
    try:
        choix = int(json.loads(r["message"]["content"])["intention"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return a                           # réponse illisible : on garde l'ensemble
    return choix if choix in (a, b) else a


def hesitantes(P, marge):
    ordre = np.argsort(-P, axis=1)
    lignes = np.arange(len(P))
    return np.where(P[lignes, ordre[:, 0]] - P[lignes, ordre[:, 1]] < marge)[0], ordre


def cmd_oof(args):
    """Mesure sur la validation croisée (réponses connues)."""
    df = load_training_frame()
    if args.qwen7b:
        # pli 0 seulement : l'ensemble 4 CamemBERT + Qwen 7B (Colab, poids 2)
        z = np.load(PROJET / "colab" / "qwen7b_fold0.npz")
        ids = z["val_ids"]
        oof = [load_oof(str(PROJET / f"cv_{g}_f0"))[0].loc[ids].values for g in ["base", "bio", "v2", "av2"]]
        P = np.average(oof + [z["val_probs"]], axis=0, weights=[1, 1, 1, 1, 2])
    else:
        oof = [load_oof(str(PROJET / f"cv_{g}_f*"))[0] for g in ["base", "bio", "v2", "av2"]]
        ids = oof[0].index
        P = np.mean([o.loc[ids].values for o in oof], axis=0)
    y = df.set_index("ID").loc[ids, "intention"].values
    q = df.set_index("ID").loc[ids, "question"].values
    idx, ordre = hesitantes(P, args.marge)
    if args.n:
        idx = np.random.default_rng(0).choice(idx, size=min(args.n, len(idx)), replace=False)
    print(f"Test sur {len(idx)} questions hésitantes (écart < {args.marge}) sur {len(P)}.\n")
    gagne = perdu = 0
    for n, k in enumerate(idx, 1):
        a, b = int(ordre[k, 0]), int(ordre[k, 1])
        choix = arbitrer(q[k], a, b, df)
        gagne += (choix == y[k]) and (a != y[k])      # Qwen corrige l'ensemble
        perdu += (choix != y[k]) and (a == y[k])      # Qwen casse une bonne réponse
        if n % 50 == 0:
            print(f"  {n:3d} : Qwen corrige {gagne}, casse {perdu}")
    net = gagne - perdu
    print(f"\nSur {len(idx)} questions : Qwen corrige {gagne} réponses, en casse {perdu} (net {net:+d}).")
    print(f"Effet sur l'accuracy globale : {net / len(idx) * len(hesitantes(P, args.marge)[0]) / len(P) * 100:+.2f} pt")


def cmd_test(args):
    """Applique l'arbitre aux questions hésitantes du test -> nouvelle soumission."""
    df = load_training_frame()
    test = pd.read_csv(PROJET / "data" / "input_test.csv")
    P = np.load(args.probs)
    idx, ordre = hesitantes(P, args.marge)
    pred = ordre[:, 0].copy()
    changes = 0
    for n, k in enumerate(idx, 1):
        a, b = int(ordre[k, 0]), int(ordre[k, 1])
        choix = arbitrer(str(test["question"].iloc[k]), a, b, df)
        changes += choix != a
        pred[k] = choix
        if n % 50 == 0:
            print(f"  {n}/{len(idx)} questions arbitrées, {changes} changements")
    pd.DataFrame({"ID": test["ID"], "intention": pred}).to_csv(args.out, index=False)
    print(f"{len(idx)} questions hésitantes, {changes} réponses changées -> {args.out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("oof"); p.add_argument("--n", type=int, default=0, help="0 = toutes")
    p.add_argument("--qwen7b", action="store_true", help="ensemble avec Qwen 7B (pli 0)")
    p.add_argument("--marge", type=float, default=0.15); p.set_defaults(func=cmd_oof)
    p = sub.add_parser("test"); p.add_argument("probs"); p.add_argument("--out", required=True)
    p.add_argument("--marge", type=float, default=0.15); p.set_defaults(func=cmd_test)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
