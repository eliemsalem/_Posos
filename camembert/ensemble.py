#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble de modèles : moyenne des probabilités.

- val : reconstruit la même validation que train.py (même seed, même découpe
  stratifiée), y évalue chaque modèle seul puis chaque combinaison.
      python ensemble.py val model_v4 model_bio model_v2tok
- test : moyenne des probabilités de test écrites par predict.py, et écrit
  submission.csv.
      python ensemble.py test submissions/a_probs.npy submissions/b_probs.npy

Les modèles doivent avoir été entraînés avec --clean (comme tous nos modèles).
"""
import argparse
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


def cmd_test(args):
    test = pd.read_csv(DATA / "input_test.csv")
    probs = np.mean([np.load(p) for p in args.probs], axis=0)
    out = pd.DataFrame({"ID": test["ID"], "intention": probs.argmax(-1)})
    out.to_csv(args.out, index=False)
    probs_path = Path(args.out).with_name(Path(args.out).stem + "_probs.npy")
    np.save(probs_path, probs)
    print(f"Ensemble de {len(args.probs)} modèles : {len(out)} prédictions écrites dans {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("val", help="évaluer modèles et combinaisons sur la validation")
    p.add_argument("models", nargs="+", help="dossiers de modèles entraînés AVEC validation")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_val)

    p = sub.add_parser("test", help="moyenner des probabilités de test -> submission.csv")
    p.add_argument("probs", nargs="+", help="fichiers *_probs.npy écrits par predict.py")
    p.add_argument("--out", default=str(HERE / "submission.csv"))
    p.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
