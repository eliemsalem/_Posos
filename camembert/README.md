# Posos — version CamemBERT

Fine-tuning de CamemBERT pour classer les questions du challenge Posos parmi
51 intentions. Score test : **0,7898**
(validation interne : accuracy 0,804, F1 macro 0,707).

## Installation

```bash
conda create -y -n posos python=3.11 && conda activate posos
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121  # GPU Pascal (GTX 10xx)
pip install -r requirements.txt
```

Sur un GPU récent, la dernière version de PyTorch convient.

## Données

Télécharger les fichiers sur le
[site du challenge](https://challengedata.ens.fr/participants/challenges/17/) et les
placer dans `data/` : `input_train.csv`, `output_train.csv`, `input_test.csv`.
`data/correction.csv` (dictionnaire de corrections orthographe/accents) est fourni.

## Utilisation

```bash
# 1. Choisir les réglages sur une validation stratifiée de 15 %
python train.py --epochs 8 --lr 5e-5 --class-weights --clean --out model_val

# 2. Réentraîner sur 100 % des données avec le meilleur nombre d'epochs
python train.py --full --epochs 6 --lr 5e-5 --class-weights --clean --out model_full

# 3. Prédire le jeu de test -> submission.csv (+ probabilités)
python predict.py --model model_full --clean
```

Environ 5 minutes par entraînement sur une GTX 1080 Ti.

| Script | Rôle |
|---|---|
| `train.py` | Fine-tuning (`--model`, `--lr`, `--epochs`, `--class-weights`, `--clean`, `--full`, …) |
| `predict.py` | Prédiction du jeu de test, format `ID,intention` attendu par la plateforme |
| `clean.py` | Correction orthographe/accents par dictionnaire, sans autre modification du texte |
| `ensemble.py` | Évaluation d'ensembles de modèles sur la validation, moyenne des probabilités de test |
| `subs.py` | Registre local des soumissions et de leurs scores ; estime le score d'une nouvelle soumission à partir de son désaccord avec les soumissions déjà notées |

## Historique des réglages

Voir l'[évolution complète 2019 → 2026](../README.md#évolution) dans le README principal.

Remarque : sur une GTX 1080 Ti (architecture Pascal, sans Tensor Cores),
l'entraînement en fp16 est environ 45 % plus lent qu'en fp32. Il est donc
désactivé par défaut (`--fp16` pour l'activer sur un GPU récent).
