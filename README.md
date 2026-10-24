# Posos — intent classification of French drug-related questions

Given a question written by a patient or a health professional about a medicine
(« Peut-on prendre du Doliprane pendant la grossesse ? »), predict which of **51 intents**
it expresses (side effects, dosage, pregnancy, contraindications, price…). Metric: accuracy.

**Result: 0.7583 in 2019 → 0.8310 in 2026** on the official test set (846 of about
1,018 public questions), **above the organisers' benchmark (0.8291, 844 questions)**, by
replacing a GRU/CNN + word2vec ensemble with fine-tuned French transformers, then adding
two 7B LLMs (Qwen2.5, Mistral) fine-tuned as classifiers.

🏆 **Challenge Data 2018 prize — best participant** of the Posos challenge
(*Predict the expected answer!*, ENS Challenge Data).

<p align="center">
  <img src="docs/attestation.jpeg" alt="Challenge Data 2018 prize certificate, best participant, Posos challenge" width="520">
</p>

## Data

The challenge data is **not included** in this repository (it is not redistributable).
Register on the [challenge page](https://challengedata.ens.fr/participants/challenges/17/)
to download it, then place the files as follows:

| File | Where | Content |
|---|---|---|
| `input_train.csv` | `camembert/data/` | `ID,question`, 8,028 training questions |
| `output_train.csv` | `camembert/data/` | `ID,intention`, labels 0–50 |
| `input_test.csv` | `camembert/data/` | `ID,question`, 2,035 test questions (no labels) |
| `correction.csv` | `camembert/data/` (included) | my hand-made dictionary of 1,861 spelling/accent fixes (2019) |

Pretrained models are downloaded from the Hugging Face Hub:
[camembert-base](https://huggingface.co/almanach/camembert-base),
[camembert-bio-base](https://huggingface.co/almanach/camembert-bio-base),
[camembertv2-base](https://huggingface.co/almanach/camembertv2-base),
[camembertav2-base](https://huggingface.co/almanach/camembertav2-base),
[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B).

## Approach

1. **A reliable evaluation first.** A single 15 % validation split (1,205 questions) turned
   out to be noisy (±1 point between folds), so model comparisons use **5-fold
   cross-validation** on all 8,028 questions (out-of-fold predictions).
2. **One change at a time**, each measured, including the ideas that failed.
3. **Error analysis** before trying new ideas: most remaining errors are confusions between
   neighbouring intents (e.g. *side effects* vs *risks*, *missed dose* vs *dosing error*),
   and a few percent of the errors look like debatable labels.
4. **Ensembles of diverse models**: averaging the class probabilities of different
   architectures gave the largest gains at the end.

## Results

### Official test score (computed by the platform on the public part of the test set)

| Submission | Score |
|---|---|
| 2019 — ensemble of 6 GRU/CNN networks on word2vec embeddings | 0.7583 |
| CamemBERT-base, tuned | 0.7898 |
| Ensemble: CamemBERT-base + CamemBERT-bio + CamemBERTv2 | 0.8124 |
| **Ensemble: 4 CamemBERT + Qwen2.5-7B QLoRA (weight 2)** | **0.8281** |
| Same + LLM arbiter on the 136 most uncertain questions (22 answers changed) | 0.8271 |
| Ensemble: 4 CamemBERT + Qwen2.5-7B (2 seeds) + Mistral-7B, QLoRA | 0.8301 |
| **Ensemble: 4 CamemBERT + Qwen2.5-7B + Mistral-7B, equal weights** (the recipe validated on fold 0) | **0.8310** |
| *Reference: organisers' benchmark (CNN)* | *0.8291* |

### Step by step (internal validation)

| Step | Validation | Notes |
|---|---|---|
| CamemBERT-base, 4 epochs, raw text | 0.624 | 15 % split |
| + 8 epochs, class-weighted loss | 0.769 | 51 classes, one class = 22 % of the data |
| + spelling/accent dictionary | 0.773 | the 2019 cleaning had also deleted words like *pendant*, *avant*, *avec* (they were in its stop-word list), which carry the intent |
| **+ learning rate 5e-5 instead of 2e-5** | **0.804** | the model was under-fitting: the single most useful change |
| Ensemble base + bio + v2 | 0.821 | 15 % split; 0.8124 on the official test |

**5-fold cross-validation (8,028 questions):**

| Model | Accuracy |
|---|---|
| CamemBERT-bio (biomedical French) | **0.7958** |
| CamemBERTv2 | 0.7871 |
| CamemBERTaV2 | 0.7854 |
| CamemBERT-base | 0.7839 |
| Ensemble of the 4 | **0.8094** |
| Ensemble of 3 (without CamemBERTaV2) | 0.8052 |

**7B LLMs fine-tuned as classifiers** (QLoRA 4-bit, LoRA r=16, 2 epochs, free Colab / Kaggle
T4), evaluated on fold 0 of the same cross-validation (1,606 questions):

| Fold 0 | Accuracy |
|---|---|
| Qwen2.5-7B alone | **0.8120** (best single model) |
| Best CamemBERT alone (CamemBERTaV2 / bio) | 0.8045 / 0.8039 |
| Ensemble of the 4 CamemBERT | 0.8188 |
| 4 CamemBERT + Qwen2.5-7B (equal weights) | 0.8288 |
| 4 CamemBERT + Qwen2.5-7B (weight 2) | 0.8362 (weight chosen on this same fold) |
| Mistral-7B-v0.3 alone | 0.8014 (agrees with Qwen on only 82.6 % of questions) |
| **4 CamemBERT + Qwen2.5-7B + Mistral-7B (equal weights)** | **0.8431** |

Mistral is weaker than Qwen on its own, but it makes different mistakes, so it adds more to
the ensemble than a heavier weight on Qwen. Two Qwen trainings with different seeds agree on
only 88.5 % of the test questions: averaging seeds reduces this noise.

### What did not work (or not enough)

| Idea | Measured effect |
|---|---|
| fp16 mixed precision on the GTX 1080 Ti (Pascal, no Tensor Cores) | training ~45 % **slower** than fp32 |
| Adding a weaker variant of the same model (CamemBERT v3 + v4) | 0.8033 vs 0.8041 alone |
| Post-hoc prior correction of the class weights | +0.25 pt, within noise |
| Label smoothing 0.1 (CamemBERT-bio, 4 folds) | −0.16 pt alone, −0.41 pt in the ensemble |
| CamemBERTaV2 alone | no better than CamemBERT-base… but +0.42 pt in the ensemble (different errors) |
| **LLM arbiter** for uncertain cases (Qwen2.5-Coder-14B via Ollama chooses between the two top intents) | +0.47 pt on the 4-CamemBERT ensemble (54 answers fixed, 16 broken, 720 cases), but only +0.12 pt once Qwen2.5-7B is in the ensemble; on the official test it cost one question (0.8281 → 0.8271) |

### The 51 intents, decoded

The intents are anonymised (numbers 0–50). Reading examples of each class gives their
meaning (my interpretation, except 1 = *other* and 39 = *multiple*, which are official):

- **Side effects**: 28 side effects (22 % of questions) · 11 what to do about a side effect ·
  27 duration of side effects · 21 stopping / withdrawal · 48 risks and danger
- **How to take it**: 42 dosage · 38 time of intake · 14 instructions for use · 35 route of
  administration · 36 preparation / dilution · 15 patches · 49 can the tablet be split
- **Intake errors**: 13 missed dose · 12 what to do after a dosing error
- **Safety**: 43 contraindications · 44 pregnancy, breastfeeding, children ·
  34 drug interactions · 33 alcohol, driving, lifestyle
- **What it is for**: 31 indications · 32 which treatment for a symptom ·
  37 mechanism of action · 22 general information
- **Timing**: 23 onset of action · 26 duration of action · 41 pharmacokinetics
- **Storage**: 7 storage · 16 after opening · 17 cold chain break · 19 expired medicine
- **The product**: 5 composition · 40 origin of ingredients · 4 therapeutic class ·
  30 dosage form · 25 available strengths · 6 packaging
- **Names and generics**: 20 brand name of a molecule · 45 generic of a drug ·
  50 generic substitution · 0 therapeutic alternative · 10 switching treatments
- **Market**: 24 availability · 3 cause of a shortage · 29 does a product exist ·
  18 marketing authorisation date
- **Administrative**: 8 dispensing rules · 9 prescription rules · 46 price · 47 reimbursement
- **Official catch-alls**: 1 other · 39 multiple questions

This explains most confusions: the dataset separates very close intents (four about storage,
five around side effects).

## Repository layout

| Path | Content |
|---|---|
| `camembert/train.py` | fine-tuning (`--model`, `--lr`, `--epochs`, `--class-weights`, `--clean`, `--folds/--fold`, `--full`, `--label-smoothing`) |
| `camembert/predict.py` | test predictions in the platform format + class probabilities |
| `camembert/clean.py` | spelling/accent correction with the dictionary, nothing else changed |
| `camembert/ensemble.py` | evaluate models and ensembles on cross-validation (`oof`), weighted average of test probabilities (`test`) |
| `camembert/subs.py` | local registry of submissions and their platform scores |
| `camembert/colab/posos_llm7b_lora.ipynb` | 7B LLM (Qwen, Mistral…) QLoRA classifier on a free Colab or Kaggle T4 (model, seed and mode as parameters) |
| `camembert/colab/posos_qwen7b_lora.ipynb` | first version, Qwen2.5-7B only |
| `camembert/colab/posos_arbitre.ipynb` | the LLM arbiter (Ollama) on a Colab T4 |
| `camembert/agent/` | side project: *Posos Arena* (Gradio demo: classify a question, play human vs model) and LLM agents that try to decode the intents (smolagents agent vs LlamaIndex workflow), plus the LLM arbiter |
| `2019_keras/posos.py` | the original 2019 code (Keras), kept as is |
| `docs/NOTE_PASSATION.md` | lab notebook (in French): every run, decision and dead end |

## How to reproduce

```bash
cd camembert
conda create -y -n posos python=3.11 && conda activate posos
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # Pascal GPU
pip install -r requirements.txt

# 1. Cross-validation of one model (repeat for --fold 0..4, and for each model)
python train.py --model almanach/camembert-bio-base --folds 5 --fold 0 --epochs 8 \
                --lr 5e-5 --class-weights --clean --out cv_bio_f0
python ensemble.py oof "cv_base_f*" "cv_bio_f*" "cv_v2_f*" "cv_av2_f*"

# 2. Final models on 100 % of the data, then test probabilities
python train.py --model almanach/camembert-bio-base --full --epochs 8 --lr 5e-5 \
                --class-weights --clean --out model_bio_full
python predict.py --model model_bio_full --clean --out bio.csv        # -> bio_probs.npy
#    (same for camembert-base 6 ep, camembertv2-base 6 ep, camembertav2-base 8 ep)

# 3. 7B LLMs: run colab/posos_llm7b_lora.ipynb with MODE = "full"
#    MODELE = "Qwen/Qwen2.5-7B", SEED = 42 -> qwen7b_full.npz ; SEED = 7 -> qwen7b_s2_full.npz
#    MODELE = "mistralai/Mistral-7B-v0.3" (gated: HF token) -> mistral7b_full.npz

# 4. Ensemble with equal weights -> submission.csv (best official score, 0.8310)
python ensemble.py test base_probs.npy bio_probs.npy v2_probs.npy av2_probs.npy \
                   qwen7b_full.npz mistral7b_full.npz
```

Training one CamemBERT takes about 5 minutes on a GTX 1080 Ti; the 7B model about
40–50 minutes on a Colab T4.

## Limitations

- The official score is computed on the public part of the test set (about 1,018 questions),
  so one question is worth about 0.1 point.
- Internal scores from 2019 and 2026 use different splits; only the official test scores
  compare directly.
- The LLMs were evaluated on fold 0 only (1,606 questions), not on all 5 folds: the
  ensemble weights rest on that single fold.
- The margin over the benchmark is two questions on the public test set; the private
  leaderboard may differ. Averaging a second Qwen seed did not help here (0.8301 vs 0.8310,
  one question: within noise).
- Several percent of the remaining errors look like ambiguous or debatable labels, which
  caps what any model can reach on this dataset.
- The intent names are my reading of the training examples, not official labels
  (except 1 and 39).
