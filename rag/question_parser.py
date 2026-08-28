"""Parser de question hybride (Principe 2 du MD, dispatch séquentiel/batch).

Déterministe pour les cas évidents (mots-clés) : c'est la majorité du
trafic, gardé rapide/gratuit/auditable, conforme à l'esprit du MD ("la
décision est déterministe... pas le modèle qui décide").

LLM léger pour les cas ambigus, définis ici comme un CONFLIT de signaux
(mots-clés list ET comparison présents simultanément -- ex: "What are all
the differences between X and Y?") -- l'absence totale de mot-clé retombe
sur le défaut déterministe "factual" (majorité du trafic selon le MD, ~80%),
pas sur un appel LLM, pour ne pas perdre l'optimisation de coût que le
Principe 2 vise explicitement. Si un usage réel montre que ce défaut est
trop souvent faux, le seuil d'ambiguïté est le seul endroit à ajuster.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

LIST_KEYWORDS = [
    "liste", "tous", "toutes", "énumère", "quels sont", "quelles sont",
    "list", "all", "enumerate", "what are", "which are",
]
COMPARISON_KEYWORDS = [
    "compare", "différence", "versus", "vs", "plus que", "moins que",
    "higher", "lower", "better", "worse", "difference between",
]

# k dynamique selon le type de question (candidats récupérés au retrieval)
K_BY_TYPE = {"factual": 3, "comparison": 7, "list": 10}

LIGHT_MODEL = "openai/gpt-oss-20b"  # "petit appel LLM" -- modele plus rapide/leger que celui de generation.py


class QuestionTypeResult(BaseModel):
    type: Literal["factual", "list", "comparison"]


def get_light_llm(model: str = LIGHT_MODEL, temperature: float = 0):
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquant dans l'environnement. "
            "Charger le .env (python-dotenv) avant d'appeler get_light_llm()."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def _deterministic_type(question: str) -> str | None:
    """None = signaux contradictoires (list ET comparison) -> ambigu, LLM requis."""
    q = question.lower()
    list_hit = any(kw in q for kw in LIST_KEYWORDS)
    comparison_hit = any(kw in q for kw in COMPARISON_KEYWORDS)
    if list_hit and comparison_hit:
        return None
    if list_hit:
        return "list"
    if comparison_hit:
        return "comparison"
    return "factual"


def parse_question_type(question: str, llm=None) -> tuple[str, str]:
    """Retourne (type, methode) avec methode in {"deterministic", "llm"}."""
    det = _deterministic_type(question)
    if det is not None:
        return det, "deterministic"

    llm = llm or get_light_llm()
    structured_llm = llm.with_structured_output(QuestionTypeResult)
    result = structured_llm.invoke(
        "Classifie cette question en exactement un type parmi : "
        "'factual' (une valeur/un fait unique attendu), "
        "'list' (plusieurs éléments attendus en réponse), "
        "'comparison' (comparaison entre au moins deux éléments).\n\n"
        f"Question : {question}"
    )
    return result.type, "llm"
