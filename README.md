# Chatbot RAG — NIST Cybersecurity Framework v1.1

Chatbot documentaire sur le NIST Cybersecurity Framework v1.1 : retrieval hybride
(keyword + embedding), answer contract avec span de preuve, dispatch
séquentiel/batch selon le type de question, résolution des renvois internes
en deux passes.

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

## Ce que le chatbot démontre

- **Retrieval hybride** : signal keyword (TF-IDF sur lemmas + petit
  dictionnaire expert pour le vocabulaire absent du corpus) combiné au
  cosine sur l'embedding `BAAI/bge-base-en-v1.5`, fusionnés par Reciprocal
  Rank Fusion pondéré (cosine prioritaire, keyword en signal correctif).
- **Answer contract typé** (Pydantic) : chaque réponse inclut un span de
  preuve exact (page + texte copié mot pour mot), et deux booléens
  (`answer_found`, `complete_answer_found`) pour ne jamais inventer une
  réponse absente du document.
- **Dispatch séquentiel/batch** selon le type de question (factual/list/
  comparison), avec un parser hybride : règles déterministes sur mots-clés,
  LLM léger seulement en cas de signaux contradictoires.
- **Résolution des renvois internes en deux passes** : quand une réponse
  pointe vers d'autres sections du document (ex: une catégorie qui renvoie
  vers ses sous-catégories), une passe supplémentaire les résout
  automatiquement et complète la réponse.

## Limites connues du prototype

- **Modèle LLM** : le plan initial visait Llama 3.3 70B via Groq, retiré du
  catalogue Groq entre-temps — remplacé par `openai/gpt-oss-120b` (le plus
  grand modèle disponible sur Groq, choisi pour la même raison : suivre
  fidèlement un schéma JSON structuré).
- **Index ChromaDB en mémoire** (`EphemeralClient`) : pas de persistance
  entre redémarrages, pas de mise à jour incrémentale. À vrai volume
  (documents multiples, mises à jour fréquentes), il faudrait passer à une
  base vectorielle persistante (Qdrant, pgvector, Chroma en mode serveur).
- **Parser de question** basé sur des mots-clés + LLM léger sur conflit
  détecté, pas un parser complet couvrant toutes les formulations possibles.
- **Résolution des renvois internes** déterministe uniquement pour les IDs
  de subcategory du document (format `XX.YY-N`) ; les références
  informatives (CIS CSC, COBIT 5, ISA 62443, ISO/IEC, NIST SP) ne sont pas
  indexées et ne peuvent donc pas être citées ni résolues.
- **Rate limit Groq (free tier)** : le nombre d'appels LLM par question est
  plafonné (max 2 en usage normal, +1 si la résolution de renvois se
  déclenche) pour rester dans les limites gratuites — au-delà de ce budget,
  une question sans réponse claire dans le document renvoie directement
  "pas trouvé" plutôt que d'insister avec plus d'appels.
