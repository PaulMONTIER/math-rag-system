# Structure du Projet MathRAG

## Vue d'ensemble

Ce document décrit l'organisation du projet MathRAG, un système de RAG (Retrieval-Augmented Generation) spécialisé pour les questions mathématiques.

## Arborescence

```
math-rag-system/
├── config/                    # Configuration du système
│   ├── config.yaml           # Configuration générale
│   └── logging_config.yaml   # Configuration des logs
│
├── data/                     # Données et caches
│   ├── raw/                  # PDFs source (ignorés par git sauf exemple)
│   ├── processed/            # Textes extraits (ignorés)
│   ├── vector_store/         # Index FAISS (ignorés)
│   ├── embeddings_cache/     # Cache des embeddings (ignorés)
│   ├── logs/                 # Logs d'exécution (ignorés)
│   └── checkpoints/          # Points de sauvegarde workflow
│
├── docs/                     # Documentation
│   ├── archive/              # Documents historiques/migration
│   ├── assets/               # Images, diagrammes
│   └── PROJECT_STRUCTURE.md  # Ce fichier
│
├── notebooks/                # Notebooks Jupyter d'analyse
│
├── scripts/                  # Scripts utilitaires
│   ├── build_vector_store.py  # Construction de l'index vectoriel
│   ├── download_pdfs.py       # Téléchargement depuis Google Drive
│   ├── setup_gdrive.py        # Configuration Google Drive API
│   ├── simple_build.py        # Build simplifié pour dev
│   ├── test_retrieval.py      # Test du système de récupération
│   │
│   ├── debug/                 # Scripts de debugging
│   │   ├── debug_context.py
│   │   ├── debug_llm_response.py
│   │   ├── debug_meta.py
│   │   ├── debug_metadata.py
│   │   └── debug_sources.py
│   │
│   └── utils/                 # Utilitaires divers
│       ├── add_langfuse_spans.py
│       ├── add_spans.py
│       ├── monitor_logs.sh
│       └── visualize_graph.py
│
├── src/                      # Code source principal
│   ├── agents/               # Agents du workflow
│   │   ├── classifier.py     # Classification d'intention
│   │   ├── planner.py        # Planification stratégie
│   │   ├── retriever.py      # Récupération documents
│   │   ├── web_searcher.py   # Recherche web
│   │   ├── generator.py      # Génération réponses
│   │   ├── editor.py         # Révision qualité
│   │   └── verifier.py       # Vérification finale
│   │
│   ├── extraction/           # Extraction de texte des PDFs
│   │   └── pdf_processor.py
│   │
│   ├── interface/            # Interface Streamlit
│   │   ├── app.py            # Application principale
│   │   ├── components/       # Composants UI réutilisables
│   │   ├── static/           # Assets statiques
│   │   └── templates/        # Templates HTML
│   │
│   ├── llm/                  # Clients LLM
│   │   ├── closed_models.py  # OpenAI, Anthropic, etc.
│   │   └── open_models.py    # Ollama (local)
│   │
│   ├── utils/                # Utilitaires système
│   │   ├── config_loader.py
│   │   ├── cost_tracker.py
│   │   ├── langfuse_context.py
│   │   ├── langfuse_integration.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   ├── vectorization/        # Système vectoriel
│   │   ├── embedder.py       # Création d'embeddings
│   │   └── vector_store.py   # Gestion index FAISS
│   │
│   └── workflow/             # Workflow LangGraph
│       └── langgraph_pipeline.py
│
├── tests/                    # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── run_test_questions.py
│   ├── test_questions.json
│   │
│   ├── test_actualites.py
│   ├── test_complete_flow.py
│   ├── test_detail_level.py
│   ├── test_fallback.py
│   ├── test_langfuse.py
│   ├── test_langfuse_tree.py
│   ├── test_manual_tracing.py
│   ├── test_questions.py
│   ├── test_quick.py
│   ├── test_simple.py
│   ├── test_simple_suggestions.py
│   ├── test_suggestions.py
│   ├── test_suggestions_debug.py
│   ├── test_validation.py
│   ├── test_web_debug.py
│   └── test_web_real.py
│
├── .env                      # Variables d'environnement (ignoré)
├── .env.example              # Template pour .env
├── .gitignore                # Fichiers ignorés par git
├── .streamlit/               # Configuration Streamlit
│   └── config.toml
│
├── Makefile                  # Commandes make
├── QUICKSTART.md             # Guide de démarrage rapide
├── README.md                 # Documentation principale
├── TESTING.md                # Guide de tests
├── requirements.txt          # Dépendances Python
└── requirements-dev.txt      # Dépendances de développement
```

## Organisation par Fonctionnalité

### Workflow Multi-Agent (LangGraph)

Le cœur du système est un graphe d'agents orchestrés par LangGraph :

1. **Classifier** → Détecte l'intention (question math, off-topic, etc.)
2. **Planner** → Choisit la stratégie (RAG local, web search, hybride)
3. **Retriever** → Récupère documents pertinents
4. **Generator** → Génère la réponse
5. **Editor** → Révise et améliore
6. **Verifier** → Vérifie la qualité
7. **Human-in-the-Loop** → Point d'approbation optionnel

### Observabilité

- **Langfuse** : Traçage manuel des appels LLM avec spans hiérarchiques
- **Métriques** : Tracking des performances et coûts
- **Logs** : Système de logging structuré

### Modes d'Exécution

1. **Local (Ollama)** : Modèles open-source locaux (Mistral, Llama, etc.)
2. **Cloud (OpenAI)** : GPT-4, GPT-4-turbo, etc.
3. **Hybride** : Combine les deux selon le contexte

## Fichiers Sensibles (Ignorés par Git)

```
.env                     # Clés API (OpenAI, Langfuse, etc.)
credentials.json         # Credentials Google Drive API
token.json               # Token OAuth2 Google
token.pickle             # Token pickle Google
data/raw/*.pdf           # PDFs sources
data/processed/          # Textes extraits
data/vector_store/       # Index FAISS
data/embeddings_cache/   # Cache embeddings
data/logs/              # Logs d'exécution
```

## Points d'Entrée

### Interface Web
```bash
streamlit run src/interface/app.py
```

### Tests
```bash
# Test complet du workflow
python tests/test_manual_tracing.py

# Tests rapides
python tests/test_quick.py

# Tests spécifiques
python tests/test_langfuse.py
python tests/test_validation.py
```

### Scripts Utilitaires
```bash
# Build vector store
python scripts/build_vector_store.py

# Download PDFs from Google Drive
python scripts/download_pdfs.py

# Setup Google Drive
python scripts/setup_gdrive.py
```

## Configuration

Voir [config/config.yaml](../config/config.yaml) pour la configuration complète :

- Modèles LLM (local/cloud)
- Paramètres de recherche vectorielle
- Limites et timeouts
- Niveaux de logging

## Documentation

- [README.md](../README.md) - Vue d'ensemble du projet
- [QUICKSTART.md](../QUICKSTART.md) - Guide de démarrage rapide
- [TESTING.md](../TESTING.md) - Guide des tests
- [docs/archive/](archive/) - Documentation historique

## Maintenance

### Nettoyage
```bash
# Nettoyer caches Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Nettoyer logs
rm -rf data/logs/*.log

# Reconstruire vector store
python scripts/build_vector_store.py --force
```

### Mise à jour des dépendances
```bash
pip install -r requirements.txt --upgrade
```

## Architecture Technique

### Stack
- **Python 3.11+**
- **LangGraph** : Orchestration multi-agent
- **FAISS** : Recherche vectorielle
- **Sentence Transformers** : Embeddings
- **Streamlit** : Interface web
- **Langfuse** : Observabilité LLM

### Design Patterns
- **Agent Pattern** : Agents spécialisés modulaires
- **Strategy Pattern** : Choix dynamique local/web/hybride
- **Observer Pattern** : Métriques et logging
- **State Pattern** : Gestion d'état workflow LangGraph
