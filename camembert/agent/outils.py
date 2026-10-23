#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils smolagents pour enquêter sur les 51 intentions anonymisées du challenge Posos.

Tout reste en local : données du challenge, ensemble CamemBERT (sur CPU) et LLM via Ollama.
"""
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from smolagents import tool

PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJET))
from clean import load_correction_map, make_corrector  # noqa: E402

# Les trois modèles de la meilleure soumission (0,8124)
MODELES = ["model_v4_full", "model_bio_full", "model_v2tok_full"]
# Modèles entraînés avec validation : leurs val_probs.npy servent à trouver les confusions
MODELES_VAL = ["model_v4", "model_bio", "model_v2tok"]


@lru_cache(maxsize=1)
def donnees():
    x = pd.read_csv(PROJET / "data" / "input_train.csv")
    y = pd.read_csv(PROJET / "data" / "output_train.csv")
    df = x.merge(y, on="ID")
    df["question"] = df["question"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def predictions_validation():
    """Questions de validation (même découpe que train.py) avec la prédiction de l'ensemble."""
    from sklearn.model_selection import train_test_split
    df = donnees()
    _, val = train_test_split(df, test_size=0.15, random_state=42, stratify=df["intention"])
    probs = np.mean([np.load(PROJET / m / "val_probs.npy") for m in MODELES_VAL], axis=0)
    val = val.copy()
    val["prediction"] = probs.argmax(-1)
    return val


@lru_cache(maxsize=1)
def ensemble():
    """Charge les trois CamemBERT une seule fois (CPU)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return [(AutoTokenizer.from_pretrained(PROJET / m),
             AutoModelForSequenceClassification.from_pretrained(PROJET / m).eval())
            for m in MODELES]


@tool
def exemples_intention(intention: int, n: int = 10) -> str:
    """Renvoie des questions d'entraînement tirées au hasard pour une intention, et sa fréquence.

    Args:
        intention: numéro de l'intention, de 0 à 50.
        n: nombre de questions à renvoyer (10 par défaut, 25 maximum).
    """
    df = donnees()
    sous = df[df["intention"] == intention]
    if sous.empty:
        return f"Intention {intention} inconnue (valeurs possibles : 0 à 50)."
    n = min(int(n), 25, len(sous))
    tirage = sous.sample(n, random_state=None)["question"].str.slice(0, 200)
    lignes = "\n".join(f"- {q}" for q in tirage)
    return (f"Intention {intention} : {len(sous)} questions "
            f"({len(sous) / len(df):.1%} du jeu d'entraînement). Exemples :\n{lignes}")


@tool
def confusions_intention(intention: int) -> str:
    """Indique avec quelles autres intentions le modèle confond celle-ci, avec des exemples
    de questions mal classées. Utile pour distinguer deux intentions proches.

    Args:
        intention: numéro de l'intention, de 0 à 50.
    """
    val = predictions_validation()
    vraies = val[val["intention"] == intention]
    if vraies.empty:
        return f"Pas de question de validation pour l'intention {intention}."
    erreurs = vraies[vraies["prediction"] != intention]
    texte = [f"Intention {intention} : {len(vraies) - len(erreurs)}/{len(vraies)} questions "
             "de validation bien classées."]
    for autre, nb in Counter(erreurs["prediction"]).most_common(3):
        exemples = erreurs[erreurs["prediction"] == autre]["question"].str.slice(0, 160).head(2)
        texte.append(f"Prise {nb} fois pour l'intention {autre}, par exemple :")
        texte += [f"  - {q}" for q in exemples]
    # dans l'autre sens : questions d'autres intentions prises pour celle-ci
    faux = val[(val["prediction"] == intention) & (val["intention"] != intention)]
    if len(faux):
        principales = ", ".join(f"{a} ({nb} fois)" for a, nb in Counter(faux["intention"]).most_common(3))
        texte.append(f"À l'inverse, le modèle prédit {intention} à tort pour : {principales}.")
    return "\n".join(texte)


@tool
def classer_question(question: str) -> str:
    """Classe une question en français avec l'ensemble de trois CamemBERT (score 0,8124 au
    challenge) et renvoie les trois intentions les plus probables avec leur probabilité.

    Args:
        question: la question à classer, en français.
    """
    import torch
    texte = make_corrector(load_correction_map())(question)
    probs = []
    for tokenizer, modele in ensemble():
        enc = tokenizer(texte, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            p = modele(**enc).logits.softmax(-1)[0].numpy()
        # colonne j = intention j (voir id2label dans config.json)
        ordre = [i for i, _ in sorted(modele.config.id2label.items(), key=lambda kv: int(kv[1]))]
        probs.append(p[ordre])
    p = np.mean(probs, axis=0)
    top = np.argsort(p)[::-1][:3]
    return "\n".join(f"intention {int(i)} : {p[i]:.0%}" for i in top)
