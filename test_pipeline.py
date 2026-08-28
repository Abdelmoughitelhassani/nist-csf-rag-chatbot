from dotenv import load_dotenv
load_dotenv()  # charge GROQ_API_KEY depuis .env

import fitz
from rag.parser import parse_nist_csf_structure
from rag.chunking import build_contextual_chunks, build_category_chunks
from rag.retrieval import HybridRetriever
from rag.generation import get_llm
from rag.orchestrator import Orchestrator
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Chargement
doc = fitz.open("data/nist_csf.pdf")
sections, category_info = parse_nist_csf_structure(doc)
sub_chunks = build_contextual_chunks(sections)
cat_chunks = build_category_chunks(sections, category_info)
all_chunks = sub_chunks + cat_chunks

# ChromaDB
client = chromadb.Client()
bge_fn = SentenceTransformerEmbeddingFunction("BAAI/bge-base-en-v1.5")
collection = client.create_collection("test", embedding_function=bge_fn)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    documents=[c["text"] for c in all_chunks],
    metadatas=[{"page": c["page"], "level": c["level"]} for c in all_chunks]
)

# Pipeline
retriever = HybridRetriever([c["text"] for c in all_chunks], collection)
llm = get_llm()
orchestrator = Orchestrator(all_chunks, retriever, llm)

# Test
questions = [
    "What backup practices keep data available after a ransomware attack?",
    "What are all the subcategories in PR.IP?",
    "What is the difference between PR.IP-4 and PR.IP-9?",
    "What does the Asset Management category cover?",
    "What is the capital of France?",  # hors document
]

for q in questions:
    print(f"\nQuestion : {q}")
    result = orchestrator.answer(q)
    answer = result['answer']  # c'est un objet Pydantic
    
    print(f"Type : {result['question_type']} ({result['classification_method']})")
    print(f"k utilisé : {result['k']}")
    print(f"answer_found : {answer.answer_found}")        # point, pas crochet
    print(f"confidence : {answer.confidence}")
    
    if answer.answer_found:
        print(f"Réponse : {answer.value}")
        if answer.evidence:
            print(f"Source : page {answer.evidence[0].page}")
            print(f"Span : {answer.evidence[0].text}")
    else:
        print("Je n'ai pas trouvé de réponse dans le document.")
    
    if result['reference_resolution']['triggered']:
        print(f"Renvois résolus : {result['reference_resolution'].get('resolved', [])}")
    
    print("-" * 50)

    