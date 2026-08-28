"""Interface Streamlit du chatbot RAG NIST CSF.

Assemble rag/parser.py, rag/chunking.py, rag/retrieval.py, rag/generation.py,
rag/question_parser.py et rag/orchestrator.py -- toute la logique a ete
construite et validee etape par etape dans notebooks/01, 02 et 03.

ChromaDB est construit en memoire (chromadb.EphemeralClient) au demarrage,
directement depuis data/nist_csf.pdf : pas de persistance entre sessions,
conforme au MD ("Vector DB : ChromaDB local, en memoire").
"""

import logging
import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Les deux lignes ci-dessus visent les warnings internes de `transformers`,
# mais la spam de tracebacks "ModuleNotFoundError: No module named
# 'torchvision'" vient d'ailleurs : le file-watcher de Streamlit (hot-reload)
# scanne recursivement les sous-modules de `transformers` (importe via
# sentence-transformers) pour suivre leurs chemins de fichiers, et log
# chaque echec d'import optionnel via SON PROPRE logger -- pas celui de
# transformers (verifie dans site-packages/streamlit/watcher/
# local_sources_watcher.py : `_LOGGER = get_logger(__name__)`, nom
# "streamlit.watcher.local_sources_watcher"). On n'a pas besoin de
# torchvision (aucune fonctionnalite vision utilisee ici), donc ces echecs
# sont sans consequence -- silencer ce logger precis.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import chromadb
import fitz
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from rag.chunking import build_category_chunks, build_contextual_chunks
from rag.generation import get_llm
from rag.orchestrator import Orchestrator
from rag.parser import parse_nist_csf_structure
from rag.question_parser import get_light_llm
from rag.retrieval import HybridRetriever

load_dotenv()  # pour un run local (streamlit run app.py) avec un fichier .env

PDF_PATH = "data/nist_csf.pdf"


def _load_secret(key: str) -> str | None:
    """Variable d'environnement (.env, charge plus haut) en priorite, sinon
    st.secrets (Streamlit Cloud). L'environnement passe en premier expres :
    .streamlit/secrets.toml livre dans ce repo n'est qu'un TEMPLATE avec des
    valeurs factices (voir le fichier) -- s'il passait en premier, il
    masquerait la vraie cle du .env en local. st.secrets peut lever une
    exception s'il n'y a aucun secrets.toml du tout -- on l'ignore alors."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets.get(key)
    except Exception:
        return None


@st.cache_resource(show_spinner="Chargement du document et des modèles (une seule fois par session serveur)...")
def load_orchestrator() -> Orchestrator:
    groq_key = _load_secret("GROQ_API_KEY")
    hf_token = _load_secret("HF_TOKEN")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token  # accelere/authentifie le telechargement du modele BGE

    doc = fitz.open(PDF_PATH)
    sections, category_info = parse_nist_csf_structure(doc)
    chunks = build_contextual_chunks(sections) + build_category_chunks(sections, category_info)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5", device="cpu",
    )
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="nist_csf", embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
    )

    retriever = HybridRetriever([c["text"] for c in chunks], collection)
    llm = get_llm()
    light_llm = get_light_llm()
    return Orchestrator(chunks, retriever, llm, light_llm)


st.set_page_config(page_title="NIST CSF Chatbot", page_icon="🔒")

st.title("🔒 NIST CSF Chatbot")
st.write(
    "Chatbot RAG documentaire sur le NIST Cybersecurity Framework v1.1. "
    "Retrieval hybride (keyword + embedding), answer contract avec span de preuve, "
    "dispatch séquentiel/batch selon le type de question, et résolution des renvois "
    "internes en deux passes."
)

orchestrator = load_orchestrator()  # spinner via show_spinner= du @st.cache_resource -- charge une fois a l'ouverture de la page, pas a la premiere question

question = st.text_input("Posez une question sur le NIST Cybersecurity Framework v1.1 :")

if question:
    with st.spinner("Recherche en cours..."):
        try:
            result = orchestrator.answer(question)
        except Exception as e:
            # filet de securite : rag/generation.py degrade deja gracieusement
            # les erreurs API Groq (retry + fallback answer_found=False), mais
            # on evite ici qu'un imprevu (reseau, ChromaDB, etc.) fasse
            # planter toute la page pour une seule question.
            st.error(f"Une erreur inattendue est survenue pendant le traitement de la question : {e}")
            st.stop()

    st.subheader("Type de question")
    st.write(f"**{result['question_type']}** (méthode : {result['classification_method']})")

    st.subheader("Retrieval")
    st.write(f"k = {result['k']} · passes séquentielles : {result['sequential_passes']}")

    st.subheader("Temps")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval", f"{result['timing_retrieval_s']} s")
    col2.metric("LLM", f"{result['timing_llm_s']} s")
    col3.metric("Total", f"{result['timing_total_s']} s")
    if "timing_pass1_s" in result:
        st.caption(f"Dont passe 1 : {result['timing_pass1_s']} s · passe 2 (renvois) : {result['timing_pass2_s']} s")

    answer = result["answer"]

    if not answer.answer_found:
        # regle explicite : toujours ce message si answer_found=False,
        # independamment de answer_completeness (bug corrige dans rag/generation.py)
        st.error("This question could not be answered from the document.")
    else:
        st.subheader("Réponse")
        st.write(answer.value)

        st.subheader("Source(s) / span de preuve")
        for span in answer.evidence:
            st.markdown(f"**Page {span.page}**")
            st.text(span.text)

        st.subheader("Confiance")
        st.write(f"{answer.confidence:.2f}")

    resolution = result["reference_resolution"]
    if resolution.get("triggered"):
        st.subheader("Renvois internes")
        if resolution.get("resolved"):
            st.write("✅ Résolues :", ", ".join(resolution["resolved"]))
        if resolution.get("unresolved"):
            st.write("⚠️ Non résolues :", ", ".join(resolution["unresolved"]))
        if resolution.get("note"):
            st.caption(resolution["note"])
