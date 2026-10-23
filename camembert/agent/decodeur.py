#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le décodeur des 51 intentions secrètes de Posos.

Un CodeAgent (smolagents) enquête intention par intention : il lit des exemples,
regarde avec quelles intentions le modèle la confond, teste ses hypothèses en
classant des questions qu'il invente, puis donne un nom à l'intention.

Tout est local : Ollama (qwen2.5-coder:14b) + ensemble CamemBERT sur CPU.
Le GPU doit être libre pour Ollama (pas d'entraînement en cours).

    ~/anaconda3/envs/hf-agents/bin/python agent/decodeur.py            # les 51
    ~/anaconda3/envs/hf-agents/bin/python agent/decodeur.py 43 28 44   # quelques-unes

Les résultats sont ajoutés au fur et à mesure dans agent/intentions.json :
on peut interrompre et relancer, les intentions déjà décodées sont sautées.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CamemBERT sur CPU, le GPU reste à Ollama

from smolagents import CodeAgent, LiteLLMModel

from outils import classer_question, confusions_intention, exemples_intention

# Fichier séparé de intentions.json (noms trouvés par Claude), pour comparer les deux
RESULTATS = Path(__file__).parent / "intentions_qwen.json"

# Même réglage que pour le cours : le 14B tient dans les 11 Go de la 1080 Ti
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    num_ctx=8192,
)

CONSIGNE = """Tu es un détective qui doit découvrir le sens caché d'une catégorie de questions.

Le challenge Posos classe des questions sur les médicaments en 51 intentions,
anonymisées par des numéros. Ta mission : trouver ce que signifie l'intention {i}.

Méthode :
1. Lis des exemples avec exemples_intention (au moins deux tirages, 15 questions en tout).
2. Regarde avec quelles intentions elle est confondue avec confusions_intention, et lis
   quelques exemples de ces intentions voisines pour bien faire la différence.
3. Formule une hypothèse, puis vérifie-la : invente 2 ou 3 questions qui correspondent à
   ton hypothèse et classe-les avec classer_question. Si elles ne sont pas classées en {i},
   revois ton hypothèse.
4. Termine avec final_answer et un dictionnaire Python :
   {{"nom": "nom court en français (3 à 6 mots)",
     "description": "une phrase qui précise ce qui la distingue des intentions voisines",
     "exemple": "une question typique",
     "confiance": "haute", "moyenne" ou "basse"}}
"""


def charger():
    return json.loads(RESULTATS.read_text()) if RESULTATS.exists() else {}


def main():
    cibles = [int(a) for a in sys.argv[1:]] or list(range(51))
    resultats = charger()
    agent = CodeAgent(
        tools=[exemples_intention, confusions_intention, classer_question],
        model=model,
        max_steps=10,
        additional_authorized_imports=["collections"],
    )
    for i in cibles:
        if str(i) in resultats:
            print(f"Intention {i} déjà décodée : {resultats[str(i)]['nom']}")
            continue
        print(f"\n===== Intention {i} =====")
        try:
            reponse = agent.run(CONSIGNE.format(i=i))
        except Exception as e:  # un échec ne doit pas arrêter les 50 autres
            print(f"Échec pour l'intention {i} : {e}")
            continue
        if not isinstance(reponse, dict) or "nom" not in reponse:
            print(f"Réponse inattendue pour l'intention {i} : {reponse!r}")
            continue
        resultats[str(i)] = reponse
        RESULTATS.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
        print(f"→ {i} = {reponse['nom']} ({reponse.get('confiance', '?')})")


if __name__ == "__main__":
    main()
