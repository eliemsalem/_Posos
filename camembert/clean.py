#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correction orthographique légère pour les questions Posos.

Contrairement au clean_text() de 2019 (qui mettait tout en minuscules, retirait la
ponctuation et remplaçait les chiffres), on garde ici le texte tel quel pour
CamemBERT — majuscules, ponctuation et chiffres sont des signaux utiles.

On se contente d'une chose que l'ancien code faisait bien et qui aide CamemBERT :
corriger les fautes fréquentes et surtout restaurer les accents
(ex. « anxiete » → « anxiété », « gelules » → « gélules »), via le dictionnaire
que tu avais constitué à la main.
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"


def load_correction_map(path=DATA / "correction.csv"):
    """Dictionnaire {mot_incorrect: mot_correct} depuis correction.csv."""
    df = pd.read_csv(path, sep=";")
    mapping = {}
    for bad, good in zip(df["A_corriger"], df["A_remplacer"]):
        if isinstance(bad, str) and isinstance(good, str) and bad.strip():
            mapping[bad.strip().lower()] = good.strip()
    return mapping


def make_corrector(mapping):
    """Renvoie une fonction qui corrige un texte, mot par mot, sans toucher
    à la casse des mots non concernés ni à la ponctuation."""
    if not mapping:
        return lambda text: text
    # un seul passage regex : on remplace chaque mot connu (insensible à la casse)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def correct(text):
        if not isinstance(text, str):
            return ""
        return pattern.sub(lambda m: mapping[m.group(0).lower()], text)

    return correct
