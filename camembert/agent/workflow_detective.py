#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le détective en version guidée : un Workflow LlamaIndex aux étapes imposées,
à comparer avec l'agent autonome de decodeur.py (smolagents).

    preuves (Python) -> hypothèse (LLM) -> vérification (CamemBERT)
        -> correction (LLM, 2 fois max si moins de 3 tests sur 4 réussis) -> conclusion

Tout est local : Ollama (qwen2.5-coder:14b) + ensemble CamemBERT sur CPU.

    ~/anaconda3/envs/hf-agents/bin/python agent/workflow_detective.py 43 28 12 50 17
"""
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CamemBERT sur CPU, le GPU reste à Ollama

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.ollama import Ollama

from outils import classer_question, donnees, predictions_validation

RESULTATS = Path(__file__).parent / "intentions_workflow.json"
SEUIL = 3          # tests réussis (sur 4) pour accepter l'hypothèse
MAX_CORRECTIONS = 2

llm = Ollama(model="qwen2.5-coder:14b", base_url="http://localhost:11434",
             temperature=0.2, request_timeout=300, context_window=8192, json_mode=True)


class Preuves(Event):
    dossier: str


class Hypothese(Event):
    hypothese: dict


class Verdict(Event):
    hypothese: dict
    resultats: list
    reussis: int


def exemples(i, n, graine=0):
    qs = donnees()[donnees()["intention"] == i]["question"]
    return [q[:150] for q in qs.sample(min(n, len(qs)), random_state=graine)]


def voisines(i, k=3):
    """Intentions les plus confondues avec i, dans les deux sens, sur la validation."""
    val = predictions_validation()
    c = Counter(val[(val["intention"] == i) & (val["prediction"] != i)]["prediction"])
    c += Counter(val[(val["prediction"] == i) & (val["intention"] != i)]["intention"])
    return [int(a) for a, _ in c.most_common(k)]


def top1(question):
    """Intention prédite par l'ensemble CamemBERT (première ligne de classer_question)."""
    return int(classer_question(question=question).splitlines()[0].split()[1])


def demander(prompt):
    reponse = llm.complete(prompt).text
    try:
        return json.loads(reponse)
    except json.JSONDecodeError:
        return {"erreur": reponse[:300]}


class DetectiveGuide(Workflow):

    @step
    async def rassembler(self, ctx: Context, ev: StartEvent) -> Preuves:
        i = ev.intention
        await ctx.store.set("intention", i)
        await ctx.store.set("corrections", 0)
        lignes = [f"INTENTION {i} ({(donnees()['intention'] == i).sum()} questions). Exemples :"]
        lignes += [f"- {q}" for q in exemples(i, 25)]
        for v in voisines(i):
            lignes.append(f"\nINTENTION VOISINE {v} (souvent confondue avec {i}). Exemples :")
            lignes += [f"- {q}" for q in exemples(v, 6, graine=1)]
        dossier = "\n".join(lignes)
        await ctx.store.set("dossier", dossier)
        return Preuves(dossier=dossier)

    @step
    async def hypothese(self, ctx: Context, ev: Preuves) -> Hypothese:
        i = await ctx.store.get("intention")
        h = demander(f"""Des questions sur les médicaments sont classées en intentions anonymes.
Voici l'intention {i} et ses intentions voisines, avec lesquelles un modèle la confond :

{ev.dossier}

Raisonne dans l'ordre des champs du JSON : d'abord ce que les questions de {i} ont en
commun (le TYPE d'information attendue, pas le médicament), puis ce qui les distingue de
chaque voisine, et seulement ensuite le nom.
Réponds en JSON :
{{"points_communs": ["3 observations sur le type d'information demandée"],
  "difference_avec_voisines": {{"<numéro voisine>": "ce qui les distingue", ...}},
  "nom": "nom précis en français (3 à 6 mots) ; interdit : un nom générique comme
          'Intention {i}', 'Questions sur les médicaments' ou 'Consultation médicaments'",
  "tests": ["4 questions INVENTÉES (ne recopie aucun exemple), courtes, typiques de {i}
             et pas des voisines"]}}""")
        return Hypothese(hypothese=h)

    @step
    async def verifier(self, ctx: Context, ev: Hypothese) -> Verdict:
        i = await ctx.store.get("intention")
        tests = ev.hypothese.get("tests", [])[:4]
        resultats = [{"question": q, "predite": top1(q)} for q in tests]
        reussis = sum(r["predite"] == i for r in resultats)
        print(f"  [{i}] « {ev.hypothese.get('nom')} » : {reussis}/{len(tests)} tests classés en {i}")
        return Verdict(hypothese=ev.hypothese, resultats=resultats, reussis=reussis)

    @step
    async def juger(self, ctx: Context, ev: Verdict) -> Hypothese | StopEvent:
        i = await ctx.store.get("intention")
        corrections = await ctx.store.get("corrections")
        if ev.reussis >= SEUIL or corrections >= MAX_CORRECTIONS:
            return StopEvent(result=await self.conclure(ctx, ev))
        await ctx.store.set("corrections", corrections + 1)
        echecs = "\n".join(f"- « {r['question']} » a été classée en {r['predite']}"
                           for r in ev.resultats if r["predite"] != i)
        # exemples des intentions où les tests sont tombés, pour voir la frontière
        tombees = sorted({r["predite"] for r in ev.resultats if r["predite"] != i})
        echecs += "".join(f"\n\nExemples de l'intention {t} :\n" + "\n".join(f"- {q}" for q in exemples(t, 5, 2))
                          for t in tombees)
        dossier = await ctx.store.get("dossier")
        h = demander(f"""Ton hypothèse pour l'intention {i} était : {json.dumps(ev.hypothese, ensure_ascii=False)}

Le modèle n'a classé que {ev.reussis} de tes questions de test en {i}. Échecs :
{echecs}

Relis le dossier et corrige ton hypothèse : qu'est-ce qui distingue vraiment {i} des intentions
où tes questions sont tombées ?

{dossier}

Réponds en JSON, même format, avec un nom DIFFÉRENT et plus précis que le précédent :
{{"points_communs": [...], "difference_avec_voisines": {{...}}, "nom": "...",
  "tests": [4 nouvelles questions inventées]}}""")
        return Hypothese(hypothese=h)

    async def conclure(self, ctx, ev):
        i = await ctx.store.get("intention")
        h = ev.hypothese
        confiance = "haute" if ev.reussis >= 3 else "moyenne" if ev.reussis == 2 else "basse"
        return {"nom": h.get("nom", "?"),
                "description": "; ".join(f"≠ {k} : {v}" for k, v in h.get("difference_avec_voisines", {}).items()),
                "exemple": (h.get("tests") or [""])[0],
                "confiance": confiance,
                "tests_reussis": f"{ev.reussis}/{len(ev.resultats)}",
                "corrections": await ctx.store.get("corrections")}


async def main():
    cibles = [int(a) for a in sys.argv[1:]] or list(range(51))
    resultats = json.loads(RESULTATS.read_text()) if RESULTATS.exists() else {}
    for i in cibles:
        print(f"\n===== Intention {i} =====")
        r = await DetectiveGuide(timeout=900).run(intention=i)
        resultats[str(i)] = r
        RESULTATS.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
        print(f"→ {i} = {r['nom']} (tests {r['tests_reussis']}, {r['corrections']} correction(s))")


if __name__ == "__main__":
    asyncio.run(main())
