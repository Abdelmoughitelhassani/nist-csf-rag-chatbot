"""Orchestrateur : dispatch séquentiel/batch (Principe 2) + boucle deux
passes pour les renvois internes (Principe 3).

Assemble rag/question_parser.py, rag/retrieval.py et rag/generation.py.

Limite assumée (documentée, pas corrigée ici) : la résolution de la passe 2
n'est déterministe que pour les renvois vers une SUBCATEGORY du document
(ex: "PR.IP-4", format `XX.YY-N`), via le registre subcategory_id -> chunk
construit à l'init. Les références informatives (CIS CSC, COBIT 5, ISA
62443, ISO/IEC, NIST SP...) ne sont pas indexées dans ChromaDB -- elles sont
même retirées du texte des chunks pendant le parsing (rag/parser.py,
clean_text_block) -- donc le LLM ne peut de toute façon pas les voir ni les
citer comme pending_reference. Un renvoi qui ne matche pas le format
subcategory_id est donc systématiquement laissé non résolu.
"""

from __future__ import annotations

import re
import time

from rag.generation import generate_answer
from rag.question_parser import K_BY_TYPE, parse_question_type
from rag.retrieval import HybridRetriever
from rag.schemas import ReferenceAwareAnswer

SUBCATEGORY_ID_RE = re.compile(r"\b(ID|PR|DE|RS|RC)\.[A-Z]{2}-\d+\b")


def _extract_subcategory_ids(raw_text: str) -> list[str]:
    """Le LLM ne met pas toujours UN id par pending_reference -- il peut
    regrouper plusieurs ids dans un seul raw_text (ex: 'voir PR.IP-1,
    PR.IP-2, ..., PR.IP-12'). On cherche tous les ids presents dans le
    texte plutot que d'exiger une correspondance exacte du champ entier."""
    return [m.group(0).upper() for m in SUBCATEGORY_ID_RE.finditer(raw_text.upper())]

MAX_SECOND_PASS = 1  # budget maximal (Principe 3 du MD) : 1 passe supplementaire par defaut

# nombre MAXIMAL de candidats essayes un par un en sequentiel, INDEPENDANT de
# k. Reduit a 1 (un seul essai avant bascule en batch) suite au rate limit
# Groq constate en usage reel : avec MAX_SEQUENTIAL=3 + le garde-fou
# d'elargissement k qui existait ici avant (desormais retire), une question
# "introuvable" pouvait declencher jusqu'a 15 appels Groq rapproches pour
# une seule question utilisateur -- assez pour heurter le rate limit du free
# tier (429, temps de reponse de plusieurs dizaines de secondes, et parfois
# des reponses corrompues). Avec MAX_SEQUENTIAL=1 et sans elargissement, le
# pire cas est desormais 2 appels LLM par question (1 essai sequentiel +
# 1 fallback batch), + 1 appel supplementaire si la passe 2 (renvois
# internes) se declenche -- largement dans le free tier.
MAX_SEQUENTIAL = 1


class Orchestrator:
    """chunks : liste combinee (subcategory + category), chaque dict avec au
    moins 'text', 'page', 'level', et 'subcategory_id' pour les chunks de
    niveau subcategory (utilise pour la resolution deterministe passe 2)."""

    def __init__(self, chunks: list[dict], retriever: HybridRetriever, llm, light_llm=None):
        self.chunks = chunks
        self.retriever = retriever
        self.llm = llm
        self.light_llm = light_llm
        self.by_subcategory_id = {
            c["subcategory_id"]: i for i, c in enumerate(chunks) if c.get("level") == "subcategory"
        }

    def _candidate(self, idx: int) -> dict:
        c = self.chunks[idx]
        return {"text": c["text"], "page": c["page"]}

    def answer(self, question: str) -> dict:
        t_start = time.perf_counter()

        # classification : mesuree a part (pas "pass1"/"pass2" du Principe 3,
        # mais un appel LLM leger qui compte dans le temps LLM total percu)
        t_classify = time.perf_counter()
        qtype, method = parse_question_type(question, llm=self.light_llm)
        classify_s = time.perf_counter() - t_classify

        k = K_BY_TYPE[qtype]
        top_idx, candidates, result, passes_used, resolution, timing = self._answer_with_k(question, qtype, k)

        total_s = time.perf_counter() - t_start
        llm_s = classify_s + timing["llm_s"]

        out = {
            "question": question,
            "question_type": qtype,
            "classification_method": method,
            "k": k,
            "candidates": [
                {"level": self.chunks[i].get("level"),
                 "id": self.chunks[i].get("subcategory_id") or self.chunks[i].get("category_code"),
                 "page": self.chunks[i]["page"]}
                for i in top_idx
            ],
            "sequential_passes": passes_used,
            "reference_resolution": resolution,
            "answer": result,
            "timing_retrieval_s": round(timing["retrieval_s"], 3),
            "timing_llm_s": round(llm_s, 3),
            "timing_total_s": round(total_s, 3),
        }
        if timing["pass2_s"] is not None:
            out["timing_pass1_s"] = round(timing["pass1_s"], 3)
            out["timing_pass2_s"] = round(timing["pass2_s"], 3)
        return out

    def _answer_with_k(self, question: str, qtype: str, k: int) -> tuple[list[int], list[dict], ReferenceAwareAnswer, int, dict, dict]:
        t_retrieval = time.perf_counter()
        top_idx = self.retriever.retrieve(question, k=k)
        retrieval_s = time.perf_counter() - t_retrieval

        candidates = [self._candidate(i) for i in top_idx]

        t_pass1 = time.perf_counter()
        if qtype == "factual":
            result, passes_used = self._sequential(question, candidates, qtype)
        else:
            result = generate_answer(question, candidates, self.llm, question_type=qtype)
            passes_used = 1
        pass1_s = time.perf_counter() - t_pass1

        result, resolution, pass2_s = self._resolve_references_if_needed(question, result, candidates, top_idx, qtype)

        timing = {
            "retrieval_s": retrieval_s,
            "llm_s": pass1_s + (pass2_s or 0.0),
            "pass1_s": pass1_s,
            "pass2_s": pass2_s,
        }
        return top_idx, candidates, result, passes_used, resolution, timing

    def _sequential(self, question: str, candidates: list[dict], qtype: str) -> tuple[ReferenceAwareAnswer, int]:
        """Mode sequentiel (factual) : top-1 d'abord, candidat suivant si la
        reponse n'est pas trouvee/complete, plafonne a MAX_SEQUENTIAL essais
        (pas len(candidates) -- voir sa docstring). Garde-fou : bascule en
        batch (tous les candidats recuperes, pas seulement les
        MAX_SEQUENTIAL premiers) apres epuisement des essais sequentiels."""
        result = None
        for i, cand in enumerate(candidates[:MAX_SEQUENTIAL], start=1):
            result = generate_answer(question, [cand], self.llm, question_type=qtype)
            if result.answer_found and result.complete_answer_found:
                return result, i
        # garde-fou : aucun candidat seul n'a suffi -> tente en batch avec tout le monde
        result = generate_answer(question, candidates, self.llm, question_type=qtype)
        return result, min(len(candidates), MAX_SEQUENTIAL) + 1

    def _resolve_references_if_needed(self, question: str, result: ReferenceAwareAnswer, original_candidates: list[dict], original_idx: list[int], qtype: str) -> tuple[ReferenceAwareAnswer, dict, float | None]:
        if result.answer_completeness != "references_unresolved" or not result.pending_references:
            return result, {"triggered": False}, None

        already_present = set(original_idx)  # evite d'ajouter en double un chunk deja dans le pool initial
        resolved_ids, unresolved = [], []
        extra_candidates = []
        for ref in result.pending_references:
            ids_found = _extract_subcategory_ids(ref.raw_text)
            if not ids_found:
                unresolved.append(ref.raw_text)
                continue
            for key in ids_found:
                if key not in self.by_subcategory_id:
                    unresolved.append(key)
                    continue
                idx = self.by_subcategory_id[key]
                if idx in already_present or key in resolved_ids:
                    continue  # deja dans le pool initial ou deja ajoute pour un autre pending_reference
                extra_candidates.append(self._candidate(idx))
                resolved_ids.append(key)

        if not extra_candidates:
            note = (
                "aucune reference resolue deterministiquement (pas un ID de subcategory connu, "
                "ex: reference informative CIS/NIST/ISO -- non indexee)"
                if unresolved else
                "toutes les references citees etaient deja dans le pool initial -- rien a ajouter"
            )
            return result, {"triggered": True, "resolved": [], "unresolved": unresolved, "note": note}, None

        # passe 2, budget = MAX_SECOND_PASS (1 par defaut) : regenere avec le contexte complet
        t_pass2 = time.perf_counter()
        final = generate_answer(question, original_candidates + extra_candidates, self.llm, question_type=qtype)
        pass2_s = time.perf_counter() - t_pass2
        return final, {"triggered": True, "resolved": resolved_ids, "unresolved": unresolved}, pass2_s
