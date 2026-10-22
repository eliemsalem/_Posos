#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prédit l'intention des questions du jeu de test avec le modèle CamemBERT entraîné,
et écrit un fichier de soumission au format du challenge (ID;intention).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
DATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model"))
    ap.add_argument("--test", default=str(DATA / "input_test.csv"))
    ap.add_argument("--out", default=str(HERE / "submission.csv"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--clean", action="store_true",
                    help="même correction qu'à l'entraînement (à activer si le modèle a été entraîné avec --clean)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    df = pd.read_csv(args.test)
    df["question"] = df["question"].fillna("").astype(str)
    if args.clean:
        from clean import load_correction_map, make_corrector
        df["question"] = df["question"].map(make_corrector(load_correction_map()))
    questions = df["question"].tolist()

    preds, probs = [], []
    for i in range(0, len(questions), args.batch_size):
        batch = questions[i:i + args.batch_size]
        enc = tokenizer(batch, truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs.append(logits.softmax(dim=-1).float().cpu().numpy())
        idx = logits.argmax(dim=-1).tolist()
        # id2label renvoie l'identifiant d'intention d'origine (0-50)
        preds.extend(int(model.config.id2label[j]) for j in idx)

    out = pd.DataFrame({"ID": df["ID"], "intention": preds})
    # format exact attendu par la plateforme ENS : en-tête "ID,intention", virgules
    out.to_csv(args.out, index=False, sep=",")
    print(f"{len(out)} prédictions écrites dans {args.out}")

    # Probabilités (colonne j = intention j), pour les ensembles et subs.py
    probs = np.concatenate(probs)
    order = [i for i, _ in sorted(model.config.id2label.items(), key=lambda kv: int(kv[1]))]
    probs_path = Path(args.out).with_name(Path(args.out).stem + "_probs.npy")
    np.save(probs_path, probs[:, order])
    print(f"Probabilités écrites dans {probs_path}")


if __name__ == "__main__":
    main()
