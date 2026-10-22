#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registre des soumissions : archive chaque submission.csv avec ses réglages et son
score plateforme, pour comparer les soumissions entre elles.

On ne connaît pas les labels du test, mais deux soumissions qui ne diffèrent que
sur d % des questions ont des scores qui diffèrent d'au plus ≈ d points. Chaque
soumission déjà notée sert donc de repère pour encadrer le score d'une nouvelle
soumission avant de l'envoyer.

    python subs.py add v4_full --model model_v4_full --val 0.804 --notes "..."
    python subs.py score v4_full 0.78978
    python subs.py list
    python subs.py compare v5_ensemble          # ou un chemin vers un .csv
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SUBS = HERE / "submissions"
REGISTRY = SUBS / "registry.csv"
COLS = ["name", "date", "model", "val_accuracy", "public_score", "notes", "params"]


def load_registry():
    if REGISTRY.exists():
        return pd.read_csv(REGISTRY, dtype={"notes": str, "params": str})
    return pd.DataFrame(columns=COLS)


def save_registry(reg):
    SUBS.mkdir(exist_ok=True)
    reg.to_csv(REGISTRY, index=False)


def load_preds(name_or_path):
    """Prédictions d'une soumission archivée (par nom) ou d'un fichier csv."""
    path = Path(name_or_path)
    if not path.suffix:
        path = SUBS / f"{name_or_path}.csv"
    return pd.read_csv(path).set_index("ID")["intention"]


def cmd_add(args):
    reg = load_registry()
    if args.name in set(reg["name"]):
        raise SystemExit(f"'{args.name}' existe déjà dans le registre.")
    SUBS.mkdir(exist_ok=True)
    src = Path(args.file)
    shutil.copy(src, SUBS / f"{args.name}.csv")
    # probabilités écrites par predict.py à côté du csv, si présentes
    probs = src.with_name(src.stem + "_probs.npy")
    if probs.exists():
        shutil.copy(probs, SUBS / f"{args.name}_probs.npy")
    # réglages d'entraînement, lus dans le run_info.json écrit par train.py
    params = ""
    if args.model and (Path(args.model) / "run_info.json").exists():
        params = json.dumps(json.loads((Path(args.model) / "run_info.json").read_text())["args"])
    row = {"name": args.name, "date": date.today().isoformat(), "model": args.model or "",
           "val_accuracy": args.val, "public_score": None, "notes": args.notes, "params": params}
    save_registry(pd.concat([reg, pd.DataFrame([row])], ignore_index=True))
    print(f"Archivé : submissions/{args.name}.csv" + (" (+ probabilités)" if probs.exists() else ""))


def cmd_score(args):
    reg = load_registry()
    if args.name not in set(reg["name"]):
        raise SystemExit(f"'{args.name}' n'est pas dans le registre.")
    reg.loc[reg["name"] == args.name, "public_score"] = args.score
    save_registry(reg)
    print(f"{args.name} : score plateforme = {args.score}")


def cmd_list(args):
    reg = load_registry()
    if reg.empty:
        print("Registre vide.")
        return
    cols = ["name", "date", "val_accuracy", "public_score", "notes"]
    print(reg.sort_values("public_score", ascending=False)[cols].to_string(index=False))


def cmd_compare(args):
    new = load_preds(args.target)
    reg = load_registry()
    lows, highs = [], []
    print(f"{'référence':<20} {'désaccord':>9} {'score réf':>9}   score possible")
    for _, r in reg.iterrows():
        if r["name"] == args.target:
            continue
        ref = load_preds(r["name"]).reindex(new.index)
        d = (ref != new).mean()
        line = f"{r['name']:<20} {d:>9.1%}"
        if pd.notna(r["public_score"]):
            # Approximation : le test public est un sous-ensemble aléatoire (~la
            # moitié) du test, donc on y retrouve à peu près le même désaccord d.
            s = float(r["public_score"])
            lo, hi = max(0.0, s - d), min(1.0, s + d)
            lows.append(lo)
            highs.append(hi)
            line += f" {s:>9.4f}   [{lo:.4f} ; {hi:.4f}]"
        print(line)
    if lows:
        print(f"\nEncadrement combiné : [{max(lows):.4f} ; {min(highs):.4f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="archiver une soumission")
    p.add_argument("name")
    p.add_argument("--file", default=str(HERE / "submission.csv"))
    p.add_argument("--model", default="", help="dossier du modèle (lit run_info.json)")
    p.add_argument("--val", type=float, default=None, help="accuracy de validation interne")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("score", help="renseigner le score plateforme")
    p.add_argument("name")
    p.add_argument("score", type=float)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("list", help="afficher le registre")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("compare", help="comparer une soumission aux soumissions archivées")
    p.add_argument("target", help="nom archivé ou chemin d'un csv")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
