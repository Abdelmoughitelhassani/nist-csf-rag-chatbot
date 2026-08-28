# Chatbot RAG — NIST Cybersecurity Framework v1.1

Chatbot documentaire sur le NIST Cybersecurity Framework v1.1 : retrieval hybride
(keyword + embedding), answer contract avec span de preuve, dispatch
séquentiel/batch selon le type de question, résolution des renvois internes
en deux passes. Voir `CONTEXT_CHATBOT_RAG_v2.md` pour le contexte complet et
les notebooks `notebooks/01_*`, `02_*`, `03_*` pour la construction et la
validation étape par étape de chaque brique.

## Structure du projet

```
Chatbot_RAG/
├── app.py                   # Interface Streamlit
├── rag/
│   ├── parser.py             # Parsing PDF (structure Function > Category > Subcategory)
│   ├── chunking.py           # Chunking contextuel (subcategory + category)
│   ├── retrieval.py          # Retrieval hybride (keyword TF-IDF+expert + embedding, fusion RRF)
│   ├── question_parser.py    # Détection type de question (factual/list/comparison)
│   ├── generation.py         # Génération avec answer contract typé (Groq)
│   ├── orchestrator.py       # Dispatch séquentiel/batch + boucle renvois internes + timings
│   └── schemas.py            # Schémas Pydantic (answer contract)
├── data/
│   └── nist_csf.pdf          # Document source
├── notebooks/                 # Validation étape par étape (chunking, ChromaDB, answer contract, orchestrateur)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml           # Template (voir plus bas) -- jamais commité avec de vraies clés
└── .env                        # Clés API en local (jamais commité)
```

## Lancer en local

Prérequis : Python 3.12, une clé API Groq gratuite ([console.groq.com](https://console.groq.com)).

```bash
# 1. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Dépendances (installe aussi le modèle spaCy via son wheel officiel)
pip install -r requirements.txt

# 3. Clés API -- créer un fichier .env à la racine du projet :
```

```
GROQ_API_KEY=gsk_votre_cle_ici
HF_TOKEN=hf_votre_token_ici
```

`HF_TOKEN` est optionnel (accélère/authentifie le téléchargement du modèle
d'embedding `BAAI/bge-base-en-v1.5` depuis Hugging Face, évite le
rate-limiting des requêtes anonymes) — le reste (embeddings, chunking,
retrieval) tourne entièrement en local sans clé API.

```bash
# 4. Lancer l'interface
streamlit run app.py
```

Le premier chargement (parsing du PDF + téléchargement du modèle
d'embedding + construction de l'index ChromaDB en mémoire) prend quelques
dizaines de secondes ; il est mis en cache (`@st.cache_resource`) et ne se
refait pas entre deux questions dans la même session serveur.

## Déployer sur Streamlit Cloud

1. Pousser ce projet sur un dépôt GitHub (public ou privé — Streamlit Cloud
   a besoin d'un dépôt Git comme source, mais **le lien public final ne
   demande aucun clone** à la personne qui l'ouvre).
2. Sur [share.streamlit.io](https://share.streamlit.io), "New app" → sélectionner
   le dépôt, la branche, et `app.py` comme fichier principal.
3. Dans les settings de l'app → **Secrets**, coller :
   ```toml
   GROQ_API_KEY = "gsk_votre_cle_ici"
   HF_TOKEN = "hf_votre_token_ici"
   ```
   (même contenu que le template `.streamlit/secrets.toml` de ce repo, mais
   avec les vraies valeurs — Streamlit Cloud ne lit jamais le fichier du
   repo pour les secrets, uniquement ce qui est collé dans son dashboard).
4. Déployer. `requirements.txt` est détecté et installé automatiquement,
   y compris le modèle spaCy (via l'URL directe du wheel, pas besoin de
   `spacy download`).

## Limites connues du prototype

Voir la section "Ce à mentionner à Kezhan avec le lien" de
`CONTEXT_CHATBOT_RAG_v2.md`, et les conclusions des notebooks `01`, `02` et
`03` pour le détail de chaque décision de conception et des écarts
constatés par rapport au plan initial (changement de modèle LLM Groq,
fusion retrieval par RRF pondéré plutôt que l'union simple décrite au
départ, etc.).

- Index ChromaDB en mémoire (`EphemeralClient`) : pas de persistance entre
  redémarrages, pas de mise à jour incrémentale.
- Parser de question basé sur des mots-clés + LLM léger sur conflit, pas un
  parser complet de tous les types de questions.
- Résolution des renvois internes déterministe uniquement pour les IDs de
  subcategory du document (`XX.YY-N`) ; les références informatives
  (CIS CSC, COBIT 5, ISA 62443, ISO/IEC, NIST SP) ne sont pas indexées et ne
  peuvent donc pas être citées ni résolues.
