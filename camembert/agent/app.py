#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posos Arena : interface Gradio locale autour du modèle CamemBERT du challenge Posos.

  🔮 Devine mon intention  -- classe une question tapée librement
  🎮 Humain vs CamemBERT   -- jeu : qui reconnaît le mieux l'intention ?
  🕵️ Le détective          -- un agent smolagents (Ollama) enquête sur une intention

    ~/anaconda3/envs/hf-agents/bin/python agent/app.py      # puis http://127.0.0.1:7860
"""
import json
import os
import random
import subprocess

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CamemBERT sur CPU, le GPU reste à Ollama

import gradio as gr
import numpy as np

from outils import (PROJET, MODELES_VAL, classer_question, confusions_intention,
                    donnees, ensemble, exemples_intention, predictions_validation)

NOMS_FICHIER = PROJET / "agent" / "intentions.json"
AVERTISSEMENT = ("⚠️ Démo d'un modèle de classification, pas un avis médical. "
                 "Pour toute question sur un traitement, demandez à votre pharmacien ou médecin.")


def noms():
    """Noms trouvés par le détective (agent/intentions.json), sinon le simple numéro."""
    trouves = json.loads(NOMS_FICHIER.read_text()) if NOMS_FICHIER.exists() else {}
    return {i: (f"{i} · {trouves[str(i)]['nom']}" if str(i) in trouves else f"Intention {i}")
            for i in range(51)}


def exemple_court(i):
    """Une question d'entraînement courte, pour illustrer une intention sans son nom."""
    qs = donnees()[donnees()["intention"] == i]["question"]
    courtes = qs[qs.str.len().between(20, 90)]
    return (courtes if len(courtes) else qs).iloc[0][:90]


# ---------------------------------------------------------------- 🔮 Devine mon intention

def deviner(question):
    if not question.strip():
        return {}, ""
    import torch
    from clean import load_correction_map, make_corrector
    texte = make_corrector(load_correction_map())(question)
    probs = []
    for tokenizer, modele in ensemble():
        enc = tokenizer(texte, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            p = modele(**enc).logits.softmax(-1)[0].numpy()
        ordre = [i for i, _ in sorted(modele.config.id2label.items(), key=lambda kv: int(kv[1]))]
        probs.append(p[ordre])
    p = np.mean(probs, axis=0)
    n = noms()
    top = int(p.argmax())
    voisines = donnees()[donnees()["intention"] == top]["question"].sample(4)
    md = f"**Questions d'entraînement de la même intention ({n[top]}) :**\n\n"
    md += "\n".join(f"- {q[:160]}" for q in voisines)
    return {n[i]: float(p[i]) for i in np.argsort(p)[::-1][:5]}, md


# ---------------------------------------------------------------- 🎮 Humain vs CamemBERT

def probs_validation():
    return np.mean([np.load(PROJET / m / "val_probs.npy") for m in MODELES_VAL], axis=0)


def nouvelle_manche(etat):
    val = predictions_validation()
    probs = probs_validation()
    k = random.randrange(len(val))
    vraie = int(val["intention"].iloc[k])
    # 3 leurres difficiles : les intentions que le modèle trouve les plus plausibles
    leurres = [int(i) for i in np.argsort(probs[k])[::-1] if i != vraie][:3]
    choix = [vraie] + leurres
    random.shuffle(choix)
    n = noms()
    etiquettes = [f"{n[i]}  —  ex. « {exemple_court(i)} »" for i in choix]
    etat = {**etat, "k": k, "vraie": vraie, "choix": choix, "etiquettes": etiquettes, "jouee": False}
    question = f"### « {val['question'].iloc[k][:400]} »"
    return etat, question, gr.update(choices=etiquettes, value=None, interactive=True), "", tableau(etat)


def jouer(reponse, etat):
    if etat.get("jouee") or reponse is None:
        return etat, gr.update(), "Choisis une réponse, puis clique sur **Valider**.", tableau(etat)
    val = predictions_validation()
    choix_humain = etat["choix"][etat["etiquettes"].index(reponse)]
    choix_modele = int(val["prediction"].iloc[etat["k"]])
    vraie = etat["vraie"]
    etat = {**etat, "jouee": True, "n": etat["n"] + 1,
            "humain": etat["humain"] + (choix_humain == vraie),
            "modele": etat["modele"] + (choix_modele == vraie)}
    n = noms()
    msg = (f"{'✅' if choix_humain == vraie else '❌'} **Toi** : {n[choix_humain]}\n\n"
           f"{'✅' if choix_modele == vraie else '❌'} **CamemBERT** : {n[choix_modele]}\n\n"
           f"🎯 **Bonne réponse** : {n[vraie]}")
    return etat, gr.update(interactive=False), msg, tableau(etat)


def tableau(etat):
    if not etat.get("n"):
        return "**Score** — Toi 0 · CamemBERT 0"
    return (f"**Score après {etat['n']} manche(s)** — Toi **{etat['humain']}** · "
            f"CamemBERT **{etat['modele']}**")


# ---------------------------------------------------------------- 🕵️ Le détective

def gpu_libre(minimum_mo=9000):
    """Ollama a besoin d'environ 9-10 Go : on refuse si un entraînement occupe le GPU."""
    try:
        sortie = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout
        return int(sortie.strip().splitlines()[0]) >= minimum_mo
    except Exception:
        return False


def enqueter(intention):
    if not gpu_libre():
        yield [{"role": "assistant", "content":
                "🚧 Le GPU est occupé (entraînement en cours ?). Le détective a besoin d'environ "
                "10 Go pour Ollama : réessaie quand le GPU est libre."}]
        return
    from smolagents import CodeAgent
    from smolagents.memory import ActionStep, FinalAnswerStep
    from decodeur import CONSIGNE, RESULTATS, charger, model

    agent = CodeAgent(tools=[exemples_intention, confusions_intention, classer_question],
                      model=model, max_steps=10)
    i = int(intention)
    messages = [{"role": "user", "content": f"🕵️ Enquête sur l'intention {i}…"}]
    yield messages
    for etape in agent.run(CONSIGNE.format(i=i), stream=True):
        if isinstance(etape, ActionStep):
            if etape.code_action:
                messages.append({"role": "assistant",
                                 "content": f"**Étape {etape.step_number}**\n```python\n{etape.code_action}\n```",
                                 "metadata": {"title": f"💭 Étape {etape.step_number}"}})
            if etape.observations:
                messages.append({"role": "assistant", "content": f"```\n{etape.observations[:1500]}\n```",
                                 "metadata": {"title": "🔎 Ce que l'agent a trouvé"}})
            if etape.error:
                messages.append({"role": "assistant", "content": f"⚠️ {etape.error}"})
            yield messages
        elif isinstance(etape, FinalAnswerStep):
            r = etape.output
            if isinstance(r, dict) and "nom" in r:
                resultats = charger()
                resultats[str(i)] = r
                RESULTATS.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
                texte = (f"## 🎉 Intention {i} = **{r['nom']}**\n\n{r.get('description', '')}\n\n"
                         f"*Exemple : {r.get('exemple', '')}* — confiance : {r.get('confiance', '?')}\n\n"
                         "Résultat enregistré dans agent/intentions_qwen.json.")
            else:
                texte = f"Conclusion de l'agent : {r}"
            messages.append({"role": "assistant", "content": texte})
            yield messages


def fiche(intention):
    i = int(intention)
    return exemples_intention(intention=i, n=6) + "\n\n" + confusions_intention(intention=i)


# ---------------------------------------------------------------- interface

with gr.Blocks(title="Posos Arena", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💊 Posos Arena\nUn ensemble de 3 CamemBERT (0,8124 au challenge Posos) "
                "et un agent détective, 100 % en local.")
    gr.Markdown(AVERTISSEMENT)

    with gr.Tab("🔮 Devine mon intention"):
        q = gr.Textbox(label="Pose une question sur un médicament",
                       placeholder="Ex. : Peut-on prendre du Doliprane pendant la grossesse ?")
        gr.Examples(["Peut-on prendre du Doliprane pendant la grossesse ?",
                     "Quels sont les effets secondaires du Levothyrox ?",
                     "Peut-on conduire avec du Xanax ?",
                     "Combien de comprimés de Spasfon par jour ?"], inputs=q)
        bouton = gr.Button("Deviner", variant="primary")
        barres = gr.Label(label="Intentions les plus probables", num_top_classes=5)
        semblables = gr.Markdown()
        bouton.click(deviner, q, [barres, semblables])
        q.submit(deviner, q, [barres, semblables])

    with gr.Tab("🎮 Humain vs CamemBERT"):
        gr.Markdown("Une vraie question du challenge, 4 intentions possibles (chacune illustrée par "
                    "un exemple). Sauras-tu faire mieux que le modèle ?")
        etat = gr.State({"n": 0, "humain": 0, "modele": 0})
        score = gr.Markdown(tableau({}))
        question = gr.Markdown()
        choix = gr.Radio(label="Quelle est l'intention ?", choices=[])
        with gr.Row():
            valider = gr.Button("Valider", variant="primary")
            suivante = gr.Button("Question suivante")
        verdict = gr.Markdown()
        sorties = [etat, question, choix, verdict, score]
        suivante.click(nouvelle_manche, etat, sorties)
        valider.click(jouer, [choix, etat], [etat, choix, verdict, score])
        demo.load(nouvelle_manche, etat, sorties)

    with gr.Tab("🕵️ Le détective"):
        gr.Markdown("Choisis une intention anonyme : un agent (smolagents + qwen2.5-coder:14b via Ollama) "
                    "lit des exemples, compare avec les intentions voisines, teste ses hypothèses "
                    "avec le modèle… et lui donne un nom.")
        with gr.Row():
            numero = gr.Dropdown(choices=list(range(51)), value=43, label="Intention")
            voir = gr.Button("📂 Voir le dossier")
            lancer = gr.Button("🕵️ Lancer l'enquête", variant="primary")
        dossier = gr.Textbox(label="Dossier de l'intention", lines=12)
        enquete = gr.Chatbot(type="messages", height=500, label="Enquête en direct")
        voir.click(fiche, numero, dossier)
        lancer.click(enqueter, numero, enquete)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")   # local uniquement : les données du challenge restent privées
