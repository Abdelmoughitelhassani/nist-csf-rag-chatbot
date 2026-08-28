"""Génération avec answer contract typé, via Groq.

Le LLM est contraint (structured output / tool calling) à répondre selon le
schéma `ReferenceAwareAnswer` (rag/schemas.py) : valeur + span de preuve
exact + booléens answer_found/complete_answer_found, plutôt qu'un texte
libre. temperature=0 pour rester déterministe, important pour un contrat
de sortie structuré.

Déviation par rapport au MD : le MD spécifie "Llama 3.3 70B via Groq"
(`llama-3.3-70b-versatile`), mais ce modèle n'est plus dans le catalogue
Groq (retiré depuis la rédaction du MD -- `client.models.list()` ne renvoie
plus aucun modèle Llama). Remplacé par `openai/gpt-oss-120b`, le plus grand
modèle actuellement disponible sur Groq, choisi pour la même raison que le
MD invoquait pour Llama 3.3 70B : suivre fidèlement un schéma JSON
structuré complexe (imbriqué, avec listes de sous-objets).

Ce module ne décide PAS du mode séquentiel/batch (Principe 2 du MD) ni de
la boucle de résolution des renvois internes (Principe 3) -- il fournit
uniquement generate_answer(), brique de base que l'orchestrateur (pas
encore implémenté) appellera plusieurs fois.
"""

from __future__ import annotations

import os
import time

import groq
from langchain_groq import ChatGroq

from rag.schemas import ReferenceAwareAnswer

MAX_RETRIES = 2  # tentatives supplementaires sur erreur API Groq (rate limit, etc.)
RETRY_BACKOFF_S = 2.0

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_INSTRUCTIONS = """Tu es un assistant qui répond à des questions UNIQUEMENT à partir des passages du document fourni (NIST Cybersecurity Framework v1.1).

Règles strictes :
- N'utilise aucune connaissance extérieure au document. Si les passages fournis ne répondent pas à la question, mets answer_found=False et value=null -- n'invente jamais de réponse.
- Le champ evidence doit contenir le texte EXACT (mot pour mot) copié depuis le passage fourni, avec le numéro de page correspondant. Ne paraphrase pas dans evidence.
- complete_answer_found=False si la réponse trouvée est partielle ou dépend d'un renvoi non résolu (ex: "voir Section X").
- Si un passage renvoie explicitement vers une autre section/table/figure non fournie ici, liste-la dans pending_references et mets answer_completeness="references_unresolved".
- confidence reflète ta certitude que la réponse est correcte ET complète, entre 0 et 1."""

# Instruction additionnelle pour les questions de type "list" (Principe 2) :
# sans ça, le LLM peut repondre par une simple liste d'IDs de subcategory
# (ex: ["PR.IP-1", "PR.IP-2", ...]) au lieu du contenu utile. On veut un
# dictionnaire ID -> description exacte, copiee depuis les passages fournis.
#
# La premiere phrase (demandee explicitement) suffit a inclure le contenu,
# mais teste seule elle a produit une LISTE de chaines "ID: description"
# plutot qu'un dictionnaire {ID: description} (value: Any dans le schema
# n'impose aucune forme) -- la deuxieme phrase precise le format JSON exact
# attendu pour value, necessaire pour obtenir vraiment un dictionnaire.
LIST_TYPE_INSTRUCTION = (
    "When listing subcategories, always include the full description of each one "
    "from the provided passages, not just the ID. "
    'Format the "value" field as a JSON object mapping each subcategory ID to its '
    'exact description, e.g. {"PR.IP-1": "...", "PR.IP-2": "..."} -- not a list of strings.'
)


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0) -> ChatGroq:
    """Instancie le client Groq. Nécessite GROQ_API_KEY dans l'environnement
    (charger un .env avec python-dotenv AVANT d'appeler cette fonction)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquant dans l'environnement. "
            "Charger le .env (python-dotenv) avant d'appeler get_llm()."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def format_candidates(candidates: list[dict]) -> str:
    """candidates : liste de dicts avec au moins les cles 'text' et 'page'."""
    blocks = [f"[Passage {i} — page {c['page']}]\n{c['text']}" for i, c in enumerate(candidates, start=1)]
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    candidates: list[dict],
    llm: ChatGroq | None = None,
    question_type: str | None = None,
) -> ReferenceAwareAnswer:
    """Génère une réponse typée (answer contract) à partir d'une question et
    d'une liste de chunks candidats.

    question_type : optionnel, transmis par l'orchestrateur (Principe 2).
    Quand "list", ajoute LIST_TYPE_INSTRUCTION au prompt -- sans ça, une
    question du type "quelles sont les subcategories de X ?" tend à
    répondre par une simple liste d'IDs plutôt que par le contenu utile
    (constaté empiriquement avant ce fix). Si certaines subcategories citées
    par le document ne sont pas dans les passages fournis (candidats hors
    top-k), le LLM ne peut logiquement décrire que celles qu'il voit --
    comportement attendu, pas une erreur.

    Robustesse constatée en testant app.py avec streamlit.testing.v1.AppTest :
    le mode séquentiel (Principe 2) peut enchaîner plusieurs appels Groq
    rapprochés pour une seule question -- suffisant pour heurter le rate
    limit du free tier (429, et parfois une erreur 400 "tool call validation
    failed" au même moment, symptôme voisin). Sans retry, ça fait planter
    toute la page Streamlit pour une seule question mal servie. Ici : retry
    avec backoff sur toute erreur API Groq, puis dégradation propre
    (answer_found=False) plutôt qu'une exception non gérée -- indiscernable
    d'un vrai "pas trouvé" côté utilisateur, limite assumée du prototype."""
    llm = llm or get_llm()
    structured_llm = llm.with_structured_output(ReferenceAwareAnswer)

    instructions = SYSTEM_INSTRUCTIONS
    if question_type == "list":
        instructions = f"{SYSTEM_INSTRUCTIONS}\n- {LIST_TYPE_INSTRUCTION}"

    prompt = f"""{instructions}

Question : {question}

Passages disponibles :
{format_candidates(candidates)}
"""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = structured_llm.invoke(prompt)
            return _enforce_contract(result)
        except groq.GroqError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))

    print(f"[rag.generation] echec Groq apres {MAX_RETRIES + 1} tentatives, degradation gracieuse : {last_error}")
    return ReferenceAwareAnswer(
        value=None, evidence=[], answer_found=False, complete_answer_found=False,
        confidence=0.0, answer_completeness="partial", pending_references=[],
    )


def _enforce_contract(result: ReferenceAwareAnswer) -> ReferenceAwareAnswer:
    """Validation post-génération : le LLM ne respecte pas toujours la
    cohérence interne du contrat de lui-même. Constaté empiriquement
    (notebook 02) : answer_found=False accompagné de answer_completeness=
    "complete", qui n'a pas de sens ("complete" quoi, s'il n'y a pas de
    réponse ?). Le schéma Pydantic (Literal) n'a pas de valeur "not_found"
    dédiée -- "partial" est la moins fausse des trois options existantes
    quand il n'y a pas de réponse du tout."""
    if not result.answer_found and result.answer_completeness != "partial":
        result = result.model_copy(update={"answer_completeness": "partial"})
    return result
