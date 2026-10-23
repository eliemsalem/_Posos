#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posos challenge — classification d'intentions de questions médicales (FR).

Version modernisée : fine-tuning de CamemBERT (BERT français) à la place de
l'ancien ensemble GRU/CNN + word2vec de 2019.

Deux modes :
- par défaut : découpe train/validation stratifiée, évaluation à chaque epoch et
  conservation de la meilleure epoch (pour comparer les réglages entre eux) ;
- --full : entraînement sur 100 % des données, nombre d'epochs fixe, sans
  validation (pour la soumission finale, une fois les réglages choisis).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = Path(__file__).parent
DATA = HERE / "data"


class WeightedTrainer(Trainer):
    """Trainer qui pondère la perte par classe, pour compenser le déséquilibre
    (les classes rares pèsent plus, ce qui fait remonter le F1 macro).
    NB : la même perte pondérée sert aussi pour eval_loss."""

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (self.class_weights.to(outputs.logits.device)
                  if self.class_weights is not None else None)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight,
                                                 label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def load_training_frame():
    """Charge questions + labels, alignés par ID."""
    x = pd.read_csv(DATA / "input_train.csv")          # ID, question
    y = pd.read_csv(DATA / "output_train.csv")         # ID, intention
    df = x.merge(y, on="ID")
    df["question"] = df["question"].fillna("").astype(str)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),              # métrique du challenge
        "f1_macro": f1_score(labels, preds, average="macro"),   # tient compte du déséquilibre
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="camembert-base")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    # médiane ≈ 17 tokens, 99e centile ≈ 165 : 256 ne tronque presque rien
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=0,
                    help="validation croisée : nombre de plis (0 = découpe simple --val-size)")
    ap.add_argument("--fold", type=int, default=0,
                    help="pli utilisé comme validation, de 0 à --folds - 1")
    ap.add_argument("--out", default=str(HERE / "model"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weights", action="store_true",
                    help="pondérer la perte par classe (compense le déséquilibre)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="lissage des étiquettes (ex. 0.1) : utile si des étiquettes sont ambiguës")
    ap.add_argument("--clean", action="store_true",
                    help="appliquer la correction orthographique/accents (clean.py)")
    ap.add_argument("--full", action="store_true",
                    help="entraîner sur 100 %% des données, sans validation (soumission finale)")
    ap.add_argument("--fp16", action="store_true",
                    help="précision mixte : utile sur GPU récent, ~45 %% plus LENT sur la 1080 Ti (Pascal)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="garder les dossiers checkpoint-* (≈1,3 Go chacun) après l'entraînement")
    args = ap.parse_args()

    # Avant la création du modèle : la tête de classification est initialisée
    # aléatoirement dans from_pretrained, le seed du Trainer arrive trop tard.
    set_seed(args.seed)

    print(f"GPU disponible : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df = load_training_frame()
    if args.clean:
        from clean import load_correction_map, make_corrector
        corrector = make_corrector(load_correction_map())
        df["question"] = df["question"].map(corrector)
        print("Correction orthographique/accents appliquée.")

    # Intentions d'origine (0-50) -> indices du modèle ; id2label est stocké dans
    # config.json et permet à predict.py de revenir aux intentions d'origine.
    labels_sorted = sorted(df["intention"].unique())
    label2id = {int(lab): i for i, lab in enumerate(labels_sorted)}
    id2label = {i: int(lab) for lab, i in label2id.items()}
    num_labels = len(labels_sorted)
    df["label"] = df["intention"].map(label2id)
    print(f"{len(df)} exemples, {num_labels} classes.")

    if args.full:
        train_df, val_df = df, None
        print(f"Mode --full : entraînement sur {len(train_df)} exemples, sans validation.")
    elif args.folds:
        # Validation croisée : en lançant les plis 0 à K-1, chaque question sert
        # une fois de validation (prédictions "out-of-fold" sur tout le train).
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        train_idx, val_idx = list(skf.split(df, df["label"]))[args.fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        print(f"Pli {args.fold + 1}/{args.folds} -- Train : {len(train_df)} | Validation : {len(val_df)}")
    else:
        # Découpe stratifiée : la validation garde la même répartition de classes.
        train_df, val_df = train_test_split(
            df, test_size=args.val_size, random_state=args.seed, stratify=df["label"]
        )
        print(f"Train : {len(train_df)} | Validation : {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Pas de padding ici : DataCollatorWithPadding complète chaque batch à la
    # longueur de sa plus longue question (padding dynamique).
    def tok(batch):
        return tokenizer(batch["question"], truncation=True, max_length=args.max_len)

    cols = ["question", "label"]
    train_ds = Dataset.from_pandas(train_df[cols], preserve_index=False).map(tok, batched=True)
    val_ds = (Dataset.from_pandas(val_df[cols], preserve_index=False).map(tok, batched=True)
              if val_df is not None else None)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # En mode --full il n'y a rien pour choisir la "meilleure" epoch : on garde la dernière.
    strategy = "no" if args.full else "epoch"
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy=strategy,
        save_strategy=strategy,
        save_total_limit=1,                  # + le meilleur : 2 checkpoints max sur le disque
        load_best_model_at_end=not args.full,
        metric_for_best_model="accuracy",
        # Regroupe les questions de longueur proche : moins de padding, donc plus
        # rapide (les longueurs vont de quelques tokens à plusieurs centaines).
        group_by_length=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    class_weights = None
    if args.class_weights:
        # "balanced" : poids = n_exemples / (n_classes * n_exemples_de_la_classe)
        w = compute_class_weight("balanced", classes=np.arange(num_labels),
                                 y=train_df["label"].values)
        class_weights = torch.tensor(w, dtype=torch.float)
        print("Pondération des classes activée.")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    trainer.train()

    metrics = {}
    if not args.full:
        metrics = trainer.evaluate()
        # La meilleure epoch est choisie sur cette même validation : le score est
        # légèrement optimiste, le vrai juge reste le classement de la plateforme.
        print("\n=== Résultat sur la validation ===")
        print(f"Accuracy  : {metrics['eval_accuracy']:.4f}")
        print(f"F1 macro  : {metrics['eval_f1_macro']:.4f}")
        # Probabilités sur la validation, pour évaluer des ensembles (ensemble.py).
        # Colonne j = indice j du modèle = intention j (les 51 intentions sont présentes).
        logits = trainer.predict(val_ds).predictions
        val_probs = torch.from_numpy(logits).float().softmax(dim=-1).numpy()

    # On sauvegarde le meilleur modèle (ou le dernier en --full) pour predict.py
    out = Path(args.out)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    # Réglages + scores, pour comparer les runs sans relire les logs
    if not args.full:
        np.save(out / "val_probs.npy", val_probs)
        np.save(out / "val_ids.npy", val_df["ID"].values)
    (out / "run_info.json").write_text(json.dumps(
        {"args": vars(args), "metrics": metrics}, indent=2, ensure_ascii=False))

    if not args.keep_checkpoints:
        for ckpt in out.glob("checkpoint-*"):
            shutil.rmtree(ckpt)
    print(f"\nModèle sauvegardé dans : {out}")


if __name__ == "__main__":
    main()
