# Note de passation — Posos (CamemBERT)

> À lire en premier pour reprendre le projet dans une nouvelle conversation / fenêtre VS Code.
> Dernière mise à jour : 2026-10-23.

## Objectif
Battre le score du challenge Posos (ENS) : **classer des questions médicales FR
parmi 51 intentions** (métrique = accuracy). Mon ancien score public (pseudo
`elieS`, mai 2019) = **0,7583** (5e au classement). Cibles au-dessus : 0,7721,
0,7800, 0,7898 ; benchmark 0,8291.
Classement : https://challengedata.ens.fr/participants/challenges/17/ranking/public

⚠️ **2 soumissions par jour maximum** (fenêtre glissante de 24 h). Ne soumettre qu'après
validation (`ensemble.py`) et `subs.py compare`.

⚠️ Le score du classement est calculé par la **plateforme** sur le jeu de test
(labels non fournis). Notre validation interne (ci-dessous) n'est qu'un **indicateur** :
le juge final est la soumission de `submission.csv` sur le site.

## Environnement
- Env conda dédié : **`posos`** (ne pas utiliser `hf-agents`, qui est pour le cours).
- PyTorch **2.4.1+cu121** (les versions récentes ont abandonné le GPU Pascal / GTX 1080 Ti).
  Réinstall si besoin : `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`
- Lancer avec : `~/anaconda3/envs/posos/bin/python ...` (ou `conda activate posos`).

## Données (`data/`)
- `input_train.csv` (8028, `ID,question`), `output_train.csv` (8028, `ID,intention`, 51 classes 0-50, déséquilibrées : classe 28 = 22 %).
- `input_test.csv` (2035, IDs 8028→10062) = jeu de test du challenge.
- `correction.csv` = dico de correction orthographe/accents (1861 entrées), utilisé par `--clean`.

## Scores obtenus (accuracy / F1 macro sur validation 15 % stratifiée)
| Version | Réglages | Accuracy | F1 macro | Dossier modèle |
|---|---|---|---|---|
| v1 (baseline) | camembert-base, 4 ep., texte brut | 0,624 | 0,20 | `model/` |
| v2 | + 8 ep. + pondération classes | 0,769 | 0,612 | `model_v2/` |
| v3 | + corrections (`--clean`) | 0,773 | 0,607 | `model_v3/` |
| **v4 (meilleur actuel)** | + `--lr 5e-5` (2e-5 sous-apprenait), fp32 | **0,804** | 0,707 | `model_v4/` |

v4 : meilleure epoch = 6 (0,804), puis plateau à ~0,80. Le run prend 5,5 min (fp16
retiré : il était ~45 % plus lent sur la 1080 Ti). **Utiliser `--lr 5e-5` pour tous
les runs suivants.** Pour la soumission : réentraîner avec `--full --epochs 6`.
Fait le 2026-10-22 : `model_v4_full/` → `submission.csv` (**score plateforme : 0,78978**, contre 0,7583 en 2019 ; val interne 0,804, donc
écart val → test ≈ −1,4 pt).

**Meilleure soumission (2026-10-22) : ensemble v4_full + bio_full (8 ep) + v2tok_full (6 ep), moyenne des probas → 0,81238** (val 0,8207). Fichiers dans `submissions/ens3_v4_bio_v2*`. Modèles : `model_v4_full/`, `model_bio_full/`, `model_v2tok_full/`.

Repère historique 2019 (sur disque USB) : un seul modèle ~66,8 %, meilleur
ensemble 6 modèles ~75,1 % (val). Donc **un seul CamemBERT ≈ dépasse déjà
l'ensemble de 2019.**

## À FAIRE EN PREMIER — entraînements interrompus à relancer
La session précédente s'est arrêtée pendant ces deux runs (non terminés) :
```bash
cd ~/Documents/Hugging-Face/posos_camembert
PY=~/anaconda3/envs/posos/bin/python

# CamemBERT-bio (modèle médical FR) — le plus prometteur
$PY train.py --model almanach/camembert-bio-base --epochs 8 --class-weights --clean --out model_bio

# CamemBERTv2 (successeur généraliste)
$PY train.py --model almanach/camembertv2-base --epochs 8 --class-weights --clean --out model_v2tok
```
Chaque run ≈ 20 min sur la 1080 Ti. Comparer leur accuracy à **0,773** (v3).
⚠️ Les lancer dans la conversation où on veut recevoir les scores (les tâches en
arrière-plan ne notifient que la session qui les a lancées).

## Générer la soumission (à partir du meilleur modèle)
```bash
$PY predict.py --model model_v3 --clean   # remplacer model_v3 par le meilleur
# -> écrit submission.csv au format "ID,intention" (virgules), prêt pour la plateforme ENS
```
`--clean` doit matcher l'entraînement (si le modèle a été entraîné avec `--clean`,
il faut `--clean` ici aussi).

## Registre des soumissions (`subs.py`, dossier `submissions/`)
Archiver CHAQUE soumission envoyée, avec son score plateforme :
```bash
$PY predict.py --model <modèle> --clean          # écrit submission.csv + submission_probs.npy
$PY subs.py add <nom> --model <modèle> --val <acc_val> --notes "..."
$PY subs.py score <nom> <score_plateforme>       # après soumission
$PY subs.py list
$PY subs.py compare <nom|fichier.csv>            # AVANT de soumettre : encadre le score
```
`compare` : un désaccord de d % avec une soumission notée → score ≈ score_réf ± d.
Les `*_probs.npy` archivés servent directement à faire des ensembles.

## Leviers restants pour monter le score
1. Modèle médical (`camembert-bio-base`) / `camembertv2-base` — en cours.
2. `camembert-large` (plus lourd, réduire batch à 8).
3. Petit ensemble de 2-3 modèles (comme en 2019).
4. Réglages fins : learning rate, max-len, early stopping.

## Fichiers du projet
- `train.py` — fine-tuning (flags : `--model --epochs --batch-size --class-weights --clean --out`)
- `predict.py` — prédiction + `submission.csv`
- `clean.py` — correction orthographe/accents (dico)
- `README.md` — doc générale
- Modèles entraînés : `model/`, `model_v2/`, `model_v3/` (git-ignorés)

## Ressources sur le disque USB (`/media/elie/265C-8015/`)
- Ancien projet complet : `disk_2021_01_06/Documents/posos2019/` (code 2019, validation, soumissions).
- Embeddings (si on voulait refaire l'ancien ensemble) : `.../posos2019/embeddings/`
  (`frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin`, `wiki.fr.vec`, etc.).

## État au 2026-10-23 (après-midi)
**Soumissions prêtes pour ce soir (~23 h 15, 2/jour)** :
1. `submission.csv` = `submissions/ens5_qwen_w2.csv` : 4 CamemBERT full + Qwen2.5-7B (poids 2). Pli 0 : 0,8362.
2. `submissions/ens5_qwen_w1.csv` : idem, poids égaux. Pli 0 : 0,8288.
Ensuite : `subs.py score <nom> <score>` et mettre à jour le README GitHub (dépôt cloné dans le scratchpad de la session ; sinon `git clone git@github.com:eliemsalem/_Posos.git`).

**Validation croisée 5 plis** (`train.py --folds 5 --fold i`, `ensemble.py oof "cv_x_f*"`), sur 8 028 questions :
bio 0,7958 · v2 0,7871 · av2 0,7854 · base 0,7839 · ensemble des 4 = 0,8094.
**Qwen2.5-7B QLoRA** (Colab T4, `colab/posos_qwen7b_lora.ipynb`) : 0,8120 sur le pli 0 (meilleur modèle seul) ;
4 CamemBERT + Qwen = 0,8288 (poids 1) / 0,8362 (poids 2) sur le pli 0. Probas test : `colab/qwen7b_full.npz`.
⚠️ transformers ≥ 4.46 : `group_by_length=True` trie aussi en prédiction → `trainer.args.group_by_length = False` avant `predict` (corrigé dans le notebook).

**Essais sans gain** : lissage des étiquettes (−0,4 pt sur l'ensemble) ; arbitre LLM (`agent/arbitre.py`) :
+0,47 pt sur l'ensemble de 4 CamemBERT, mais seulement +0,12 une fois Qwen 7B ajouté → abandonné.
**Non testé** : camembert-large (arrêté pour libérer le GPU), masquage des noms de médicaments.

**À côté** : `agent/` = Posos Arena (Gradio, `python agent/app.py`), 51 intentions décodées
(`agent/intentions.json`), détectives Qwen (agent libre / workflow LlamaIndex).
Projet frère : `~/Documents/Hugging-Face/kili_data_centric` (mémo `MEMO_DEPUIS_POSOS.md`).

## Résultats du 2026-10-23 au soir
- `ens5_qwen_w2` (4 CamemBERT + Qwen2.5-7B, poids 2) : **0,82809 = 843/1018** → MEILLEUR SCORE.
  Benchmark Posos 0,8291 = 844/1018 : il manque UNE question pour l'égaler, deux pour le battre.
- `ens5_qwen_w2_arbitre` (même chose + arbitre Qwen 14B sur 136 questions hésitantes, via Colab) :
  0,82711 = 842/1018 (−1). L'arbitre n'aide plus une fois Qwen 7B dans l'ensemble.
- GitHub mis à jour (README anglais, 0,8281). Pistes restantes : 2e seed de Qwen 7B (Colab),
  camembert-large, poids de Qwen à affiner avec plus d'un pli.

## 2026-10-24 : BENCHMARK BATTU
- `ens7_qwen2_mistral` = 4 CamemBERT (1) + Qwen2.5-7B seeds 42 et 7 (0,5 + 0,5) + Mistral-7B-v0.3 (1) :
  **0,83006 = 845/1018** > benchmark Posos 0,8291 (844/1018). Meilleur score du projet.
- Mistral-7B : 0,8014 seul sur le pli 0, mais accord de 82,6 % seulement avec Qwen → ensemble
  0,8431 sur le pli 0 (contre 0,8362 sans lui). Notebooks : `colab/posos_mistral7b.ipynb`,
  `colab/posos_qwen7b_s2.ipynb` (Kaggle). Probas : `colab/*_full.npz`.
- `ens6_qwen_mistral` (4 CamemBERT + Qwen seed 42 + Mistral, poids égaux, la recette exacte du pli 0) :
  **0,83104 = 846/1018** → NOUVEAU MEILLEUR SCORE (+2 sur le benchmark).
  La moyenne des 2 seeds de Qwen (`ens7`, 845) n'a pas aidé ici : différence d'une question, du bruit.
