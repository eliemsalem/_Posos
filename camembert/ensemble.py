#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble de modèles : moyenne des probabilités.

- val : reconstruit la même validation que train.py (même seed, même découpe
  stratifiée), y évalue chaque modèle seul puis chaque combinaison.
      python ensemble.py val model_v4 model_bio model_v2tok
- oof : évalue des modèles entraînés en validation croisée (train.py --folds K
  --fold i). Chaque groupe = un motif de dossiers (un dossier par pli) ; les
  prédictions des plis couvrent tout le train, d'où une validation sur 8 028 questions.
      python ensemble.py oof "cv_av2_f*" "cv_bio_f*"
- test : moyenne des probabilités de test écrites par predict.py, et écrit
  submission.csv.
      python ensemble.py test a_probs.npy b_probs.npy qwen7b_full.npz --weights 1 1 2

Les modèles doivent avoir été entraînés avec --clean (comme tous nos modèles).
"""
import argparse
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clean import load_correction_map, make_corrector
from train import load_training_frame

HERE = Path(__file__).parent
DATA = HERE / "data"


def predict_probs(model_dir, questions, batch_size=64, max_len=256):
    """Probabilités (colonne j = intention j) d'un modèle sur une liste de questions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    for i in range(0, len(questions), batch_size):
        enc = tokenizer(questions[i:i + batch_size], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs.append(model(**enc).logits.softmax(dim=-1).float().cpu().numpy())
    order = [i for i, _ in sorted(model.config.id2label.items(), key=lambda kv: int(kv[1]))]
    return np.concatenate(probs)[:, order]


def cmd_val(args):
    df = load_training_frame()
    df["question"] = df["question"].map(make_corrector(load_correction_map()))
    # même découpe que train.py (labels 0-50 tous présents, donc label == intention)
    _, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed,
                                 stratify=df["intention"])
    y = val_df["intention"].values
    questions = val_df["question"].tolist()

    probs = {}
    for m in args.models:
        cache = Path(m) / "val_probs.npy"
        if cache.exists():
            probs[m] = np.load(cache)
        else:
            probs[m] = predict_probs(m, questions)
            np.save(cache, probs[m])

    rows = []
    for k in range(1, len(args.models) + 1):
        for combo in combinations(args.models, k):
            pred = np.mean([probs[m] for m in combo], axis=0).argmax(-1)
            rows.append({"modèles": " + ".join(combo),
                         "accuracy": accuracy_score(y, pred),
                         "f1_macro": f1_score(y, pred, average="macro"),
                         "erreurs": int((pred != y).sum())})
    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(f"Validation : {len(y)} questions (1 question = {1 / len(y):.2%})\n")
    print(res.to_string(index=False, float_format="%.4f"))


def load_oof(pattern):
    """Probabilités out-of-fold d'un groupe de dossiers (un par pli), indexées par ID."""
    dirs = sorted(Path(d) for d in glob.glob(pattern))
    if not dirs:
        raise SystemExit(f"Aucun dossier ne correspond à {pattern}")
    ids = np.concatenate([np.load(d / "val_ids.npy") for d in dirs])
    probs = np.concatenate([np.load(d / "val_probs.npy") for d in dirs])
    return pd.DataFrame(probs, index=ids).sort_index(), len(dirs)


def cmd_oof(args):
    y = load_training_frame().set_index("ID")["intention"]
    probs = {}
    for pat in args.patterns:
        df, n = load_oof(pat)
        print(f"{pat} : {n} plis, {len(df)} questions")
        probs[pat] = df
    # on ne compare que les questions couvertes par tous les groupes
    common = sorted(set.intersection(*(set(p.index) for p in probs.values())))
    y = y.loc[common].values
    rows = []
    for k in range(1, len(probs) + 1):
        for combo in combinations(probs, k):
            pred = np.mean([probs[c].loc[common].values for c in combo], axis=0).argmax(-1)
            rows.append({"modèles": " + ".join(combo),
                         "accuracy": accuracy_score(y, pred),
                         "f1_macro": f1_score(y, pred, average="macro"),
                         "erreurs": int((pred != y).sum())})
    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(f"\nValidation croisée : {len(y)} questions (1 question = {1 / len(y):.3%})\n")
    print(res.to_string(index=False, float_format="%.4f"))


def cmd_test(args):
    test = pd.read_csv(DATA / "input_test.csv")
    # .npy (predict.py) ou .npz avec une clé test_probs (notebook Colab)
    charges = [np.load(p)["test_probs"] if p.endswith(".npz") else np.load(p) for p in args.probs]
    probs = np.average(charges, axis=0, weights=args.weights)
    out = pd.DataFrame({"ID": test["ID"], "intention": probs.argmax(-1)})
    out.to_csv(args.out, index=False)
    probs_path = Path(args.out).with_name(Path(args.out).stem + "_probs.npy")
    np.save(probs_path, probs)
    print(f"Ensemble de {len(args.probs)} modèles (poids {args.weights or 'égaux'}) : "
          f"{len(out)} prédictions écrites dans {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("val", help="évaluer modèles et combinaisons sur la validation")
    p.add_argument("models", nargs="+", help="dossiers de modèles entraînés AVEC validation")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_val)

    p = sub.add_parser("oof", help="évaluer des modèles entraînés en validation croisée")
    p.add_argument("patterns", nargs="+", help='motifs de dossiers, ex. "cv_av2_f*"')
    p.set_defaults(func=cmd_oof)

    p = sub.add_parser("test", help="moyenner des probabilités de test -> submission.csv")
    p.add_argument("probs", nargs="+", help="fichiers *_probs.npy (predict.py) ou .npz (Colab)")
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="un poids par fichier (défaut : moyenne simple)")
    p.add_argument("--out", default=str(HERE / "submission.csv"))
    p.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
