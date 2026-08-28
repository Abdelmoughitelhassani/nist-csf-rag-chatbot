"""Retrieval hybride (keyword + embedding) pour le chatbot RAG NIST CSF.

Implémente le Principe 1 du MD (CONTEXT_CHATBOT_RAG_v2.md) : combiner un
signal keyword et un signal embedding cosine (via ChromaDB) en parallèle,
plutôt que de se reposer sur le cosine seul.

Version "corrigée" par rapport à la première implémentation du MD : le
signal keyword n'est plus un simple comptage de mots (`question.split()`),
mais un score TF-IDF pondéré, calculé sur des lemmas (spaCy) après
suppression de la ponctuation et des stopwords, complété par un petit
dictionnaire expert pour le vocabulaire métier absent du corpus (ex:
"ransomware" n'apparaît dans aucun chunk du NIST CSF v1.1).

Validé pas à pas dans notebooks/01_chunking_chromadb_test.ipynb (avant/après
correction du signal keyword, sur deux granularités de chunking) puis dans
notebooks/03_orchestrator.ipynb (fusion par RRF, voir plus bas).

Fusion keyword+cosine : Reciprocal Rank Fusion (RRF), pas l'union "keyword
d'abord" du MD. Découvert en ajoutant les chunks de catégorie (notebook 03) :
la fusion du MD place TOUT chunk avec un score keyword > 0, même minime,
avant TOUT chunk cosine-only -- un chunk qui est #1 en cosine peut donc être
relégué loin derrière par une quinzaine de chunks qui ne matchent que sur
1-2 mots génériques (même symptôme que la régression 1→23 déjà corrigée une
fois par le TF-IDF dans le notebook 01, mais qui revient dès que le corpus
contient assez de chunks avec un petit score keyword non-discriminant). RRF
combine les RANGS des deux classements (1/(k+rang_keyword) + 1/(k+rang_cosine))
au lieu de faire primer la présence d'un match keyword sur sa magnitude
relative au signal cosine -- standard de la littérature (Cormack et al.,
2009), voir notebooks/03_orchestrator.ipynb pour la démonstration avant/après.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Lemmatisation (spaCy) : ponctuation retirée + stopwords filtrés
# --------------------------------------------------------------------------

CUSTOM_STOPWORDS = {
    "a", "the", "what", "after", "keep", "make", "is", "are", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "from",
}

_PUNCTUATION_RE = re.compile(r"[^\w\s]")


@lru_cache(maxsize=1)
def get_nlp():
    """Charge le modele spaCy une seule fois par processus (cache)."""
    import spacy
    return spacy.load("en_core_web_sm")


def strip_punctuation(text: str) -> str:
    """Retire la ponctuation via regex avant tokenisation, ex: 'attack?' -> 'attack '."""
    return _PUNCTUATION_RE.sub(" ", text)


def lemma_tokens(text: str) -> list[str]:
    """Ponctuation retiree -> spaCy tokenise + lemmatise -> stopwords filtres.
    Retourne une LISTE (pas un set) : necessaire pour que la frequence d'un
    mot dans un chunk compte correctement dans le TF-IDF."""
    cleaned = strip_punctuation(text.lower())
    doc = get_nlp()(cleaned)
    return [
        tok.lemma_ for tok in doc
        if tok.lemma_.strip() and not tok.is_space and tok.lemma_ not in CUSTOM_STOPWORDS
    ]


def lemma_set(text: str) -> set[str]:
    return set(lemma_tokens(text))


# --------------------------------------------------------------------------
# Dictionnaire expert : vocabulaire metier absent du corpus
# --------------------------------------------------------------------------
# Les entrees contenant un "." ou un "-" sont des IDs de subcategory/categorie
# (ex "pr.ip-4", "pr.ac") : elles sont cherchees par sous-chaine dans le texte
# BRUT en minuscule du chunk (un ID ne survit pas a strip_punctuation, donc on
# le cherche avant nettoyage, sur le texte original).
EXPERT_DICT: dict[str, list[str]] = {
    "ransomware": ["backup", "recovery", "pr.ip-4", "restore"],
    "backup": ["pr.ip-4", "backup", "recovery"],
    "encrypt": ["pr.ds-1", "pr.ds-2", "encrypt"],
    "access": ["pr.ac", "identity", "authentication"],
    "incident": ["rs.rp", "respond", "recovery"],
}

EXPERT_BONUS_WEIGHT = 3.0  # poids fixe par terme lie trouve (design choice, cf. notebook 01)


def expert_bonus(question_lemmas: set[str], chunk_text_lower: str, chunk_lemmas: set[str]) -> tuple[float, list[tuple[str, str]]]:
    """Renvoie (bonus_total, liste des (trigger, terme_lie) qui ont matche)."""
    bonus = 0.0
    matched = []
    for trigger, related_terms in EXPERT_DICT.items():
        if trigger not in question_lemmas:
            continue
        for rel in related_terms:
            is_id_like = "." in rel or "-" in rel
            hit = (rel in chunk_text_lower) if is_id_like else (rel in chunk_lemmas)
            if hit:
                bonus += EXPERT_BONUS_WEIGHT
                matched.append((trigger, rel))
    return bonus, matched


# --------------------------------------------------------------------------
# Embedding (BGE) : prefixe obligatoire pour les questions, pas pour les documents
# --------------------------------------------------------------------------

def embed_query_text(question: str) -> str:
    """Prefixe obligatoire pour les questions avec BGE (pas pour les documents)."""
    return f"Represent this sentence for searching relevant passages: {question}"


# constante standard de la litterature pour le Reciprocal Rank Fusion
# (Cormack, Clarke & Buettcher, 2009) -- attenue le poids des rangs tres
# bas sans les annuler completement
RRF_K = 60

# Poids RRF : le MD decrit le cosine comme le signal PRINCIPAL et le
# keyword comme un "vote complementaire" (Principe 1) -- pas 50/50. Sans
# ponderation, le rang keyword (bruite sur des chunks longs/generiques
# comme les chunks de categorie, qui partagent du vocabulaire avec toutes
# leurs subcategories) peut faire chuter un chunk pourtant #1 en cosine
# loin dans le classement final (constate empiriquement : un chunk
# categorie #1 en cosine tombe 7e/131 en RRF non pondere). 0.7/0.3 restaure
# la primaute du cosine tout en gardant le keyword comme signal correctif.
RRF_WEIGHT_COSINE = 0.7
RRF_WEIGHT_KEYWORD = 0.3

# le bonus expert est ajoute EN PLUS du score RRF (pas fondu dans le rang
# keyword) : un rang keyword, meme #1, ne contribue jamais plus de
# 1/(RRF_K+1) ~= 0.0164 au score final -- trop faible pour que le
# dictionnaire expert reste decisif la ou aucun signal cosine/tfidf
# n'existe (ex: "ransomware" absent du corpus). EXPERT_SCORE_BOOST est
# choisi pour qu'UN SEUL terme lie matche domine largement l'etendue
# typique du score RRF (~0.033 au maximum) : 0.05 * EXPERT_BONUS_WEIGHT
# (3.0) = 0.15, sans quoi le bonus expert redevient bruit parmi d'autres.
EXPERT_SCORE_BOOST = 0.05


# --------------------------------------------------------------------------
# Retrieval hybride
# --------------------------------------------------------------------------

class HybridRetriever:
    """Retrieval hybride keyword (TF-IDF + dictionnaire expert) + embedding
    cosine (via une collection ChromaDB deja peuplee).

    Usage :
        retriever = HybridRetriever(chunk_texts, collection)
        top_idx = retriever.retrieve(question, k=5)   # -> indices dans chunk_texts

    L'index TF-IDF et les lemmas par chunk sont calcules une seule fois a la
    construction (cout amorti sur tous les appels a retrieve()), pas a
    chaque requete.
    """

    def __init__(self, chunk_texts: list[str], collection, expert_bonus_weight: float = EXPERT_BONUS_WEIGHT):
        self.chunk_texts = chunk_texts
        self.collection = collection
        self.expert_bonus_weight = expert_bonus_weight
        self._build_index()

    def _build_index(self) -> None:
        per_chunk_lemmas = [lemma_tokens(t) for t in self.chunk_texts]
        clean_corpus = [" ".join(toks) for toks in per_chunk_lemmas]
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w-]*\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(clean_corpus)
        self.chunk_lemma_sets = [set(toks) for toks in per_chunk_lemmas]
        self.chunk_texts_lower = [t.lower() for t in self.chunk_texts]

    def _tfidf_score(self, question_lemmas: set[str], idx: int) -> float:
        """Score TF-IDF pur (sans bonus expert) -- sert a calculer le RANG
        keyword utilise dans la fusion RRF."""
        row = self.tfidf_matrix[idx]
        score = 0.0
        for word in question_lemmas:
            col = self.vectorizer.vocabulary_.get(word)
            if col is not None:
                score += row[0, col]
        return score

    def _expert_bonus(self, question_lemmas: set[str], idx: int) -> float:
        bonus, _ = expert_bonus(question_lemmas, self.chunk_texts_lower[idx], self.chunk_lemma_sets[idx])
        return bonus

    def keyword_score(self, question_lemmas: set[str], idx: int) -> float:
        """Score keyword complet (TF-IDF + bonus expert) -- pour diagnostic
        / affichage. La fusion RRF (voir full_rank) traite les deux
        composantes separement : le TF-IDF determine un RANG, le bonus
        expert est ajoute en score brut apres la fusion (cf. EXPERT_SCORE_BOOST)."""
        return self._tfidf_score(question_lemmas, idx) + self._expert_bonus(question_lemmas, idx)

    def _keyword_ranking(self, question_lemmas: set[str]) -> list[tuple[int, float]]:
        scores = [(i, self._tfidf_score(question_lemmas, i)) for i in range(len(self.chunk_texts))]
        scores.sort(key=lambda x: -x[1])
        return scores

    def _cosine_ranking(self, question: str) -> list[int]:
        """Classement cosine complet (tous les chunks), pas juste top-k --
        necessaire pour calculer un rang RRF pour chaque chunk."""
        n = len(self.chunk_texts)
        results = self.collection.query(query_texts=[embed_query_text(question)], n_results=n)
        return [int(id_.split("_")[1]) for id_ in results["ids"][0]]

    def full_rank(self, question: str) -> list[int]:
        """Classement complet fusionne (tous les chunks, du plus au moins
        pertinent) : RRF(rang TF-IDF, rang cosine) + bonus expert ajoute en
        score brut (dominant si present, cf. EXPERT_SCORE_BOOST).
        retrieve(k) = full_rank(question)[:k]."""
        n = len(self.chunk_texts)
        question_lemmas = lemma_set(question)
        keyword_rank = {i: r for r, (i, _score) in enumerate(self._keyword_ranking(question_lemmas), start=1)}
        cosine_rank = {i: r for r, i in enumerate(self._cosine_ranking(question), start=1)}

        def final_score(i: int) -> float:
            rrf = (
                RRF_WEIGHT_COSINE / (RRF_K + cosine_rank[i])
                + RRF_WEIGHT_KEYWORD / (RRF_K + keyword_rank[i])
            )
            return rrf + EXPERT_SCORE_BOOST * self._expert_bonus(question_lemmas, i)

        return sorted(range(n), key=lambda i: -final_score(i))

    def retrieve(self, question: str, k: int = 5) -> list[int]:
        """Top-k indices, fusion keyword+cosine par Reciprocal Rank Fusion
        (voir docstring du module)."""
        return self.full_rank(question)[:k]
