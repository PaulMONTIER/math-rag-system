# MathRAG - Système RAG Multi-Agent pour Mathématiques

> Un système de Retrieval-Augmented Generation (RAG) intelligent spécialisé dans les questions mathématiques, utilisant LangGraph pour orchestrer plusieurs agents IA.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## À Propos

MathRAG est un système de question-réponse intelligent conçu pour répondre à des questions mathématiques complexes en combinant :

- **Recherche vectorielle** dans une base de documents mathématiques (PDFs de cours)
- **Recherche web** pour les informations récentes ou non disponibles localement
- **Orchestration multi-agent** avec LangGraph pour un workflow intelligent
- **Observabilité LLM** avec Langfuse pour tracer et optimiser les performances
- **Interface web intuitive** avec Streamlit et support LaTeX

Le système décide automatiquement de la meilleure stratégie (RAG local, web search, ou hybride) en fonction de la question posée.

## Fonctionnalités

### Intelligence Multi-Agent

- **Classifier** : Détecte l'intention de la question (mathématique, actualités, off-topic)
- **Planner** : Choisit la stratégie optimale (RAG local, web, ou hybride)
- **Retriever** : Récupère les documents pertinents via recherche vectorielle FAISS
- **Web Searcher** : Recherche sur le web pour informations récentes ou complémentaires
- **Generator** : Génère une réponse détaillée avec citations des sources
- **Editor** : Révise et améliore la qualité de la réponse
- **Verifier** : Vérifie la complétude et propose des suggestions de suivi

### Modes d'Exécution

- **Local** : Utilise Ollama avec modèles open-source (Mistral, Llama, etc.)
- **Cloud** : Utilise OpenAI GPT-4 / GPT-4-turbo
- **Hybride** : Combine les deux selon les besoins

### Observabilité et Monitoring

- **Langfuse** : Traçage complet des appels LLM avec spans hiérarchiques
- **Métriques** : Tracking des performances, latences, et coûts
- **Logs structurés** : Système de logging détaillé pour debugging

### Interface Utilisateur

- **Interface web Streamlit** avec support LaTeX complet
- **Affichage des sources** avec métadonnées (page, section)
- **Suggestions de questions** de suivi intelligentes
- **Historique** des conversations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Interface Streamlit                     │
│                    (Port 8501 par défaut)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow                         │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐              │
│  │Classifier├───►│ Planner  ├───►│ Retriever  │              │
│  └──────────┘   └──────────┘   └─────┬──────┘              │
│                                       │                       │
│  ┌──────────┐   ┌──────────┐   ┌────▼──────┐               │
│  │ Verifier │◄──┤  Editor  │◄──┤ Generator │               │
│  └──────────┘   └──────────┘   └────┬──────┘               │
│                                       │                       │
│                                  ┌────▼────────┐             │
│                                  │ Web Searcher│             │
│                                  └─────────────┘             │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐   ┌──────────┐
   │  FAISS  │    │   LLM    │   │ Langfuse │
   │ Vector  │    │  (Local  │   │  Traces  │
   │  Store  │    │or Cloud) │   │          │
   └─────────┘    └──────────┘   └──────────┘
```

## Technologies

- **Python 3.11+** : Langage principal
- **LangGraph** : Orchestration du workflow multi-agent
- **LangChain** : Framework pour LLM et RAG
- **FAISS** : Recherche vectorielle haute performance
- **Sentence Transformers** : Génération d'embeddings
- **Streamlit** : Interface web interactive
- **Langfuse** : Observabilité et tracing LLM
- **OpenAI API** : GPT-4 / GPT-4-turbo (optionnel)
- **Ollama** : Modèles locaux open-source (optionnel)

## Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) Ollama installé pour mode local
- (Optionnel) Clés API OpenAI pour mode cloud
- (Optionnel) Clés API Langfuse pour observabilité

## Installation

### 1. Cloner le Dépôt

```bash
git clone https://github.com/PaulMONTIER/math-rag-system.git
cd math-rag-system
```

### 2. Créer un Environnement Virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration

Créer un fichier `.env` à la racine du projet (copier depuis `.env.example`) :

```bash
# Configuration LLM
LLM_MODE=local  # ou 'cloud' ou 'hybrid'
OPENAI_API_KEY=your_openai_api_key_here  # Si mode cloud

# Configuration Langfuse (optionnel)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Configuration Google Drive (optionnel)
# Pour télécharger les PDFs depuis Google Drive
```

### 5. Préparer les Données

#### Option A : Télécharger depuis Google Drive (si configuré)

```bash
# Configurer l'API Google Drive (première fois uniquement)
python scripts/setup_gdrive.py

# Télécharger les PDFs
python scripts/download_pdfs.py
```

#### Option B : Ajouter vos PDFs manuellement

Placer vos fichiers PDF dans le dossier `data/raw/`.

### 6. Construire l'Index Vectoriel

```bash
python scripts/build_vector_store.py
```

Cette étape :
- Extrait le texte des PDFs
- Génère les embeddings
- Construit l'index FAISS
- Peut prendre 10-30 minutes selon le nombre de documents

## Utilisation

### Lancer l'Interface Web

```bash
streamlit run src/interface/app.py
```

L'interface sera accessible à : `http://localhost:8501`

### Utilisation Programmatique

```python
from src.workflow.langgraph_pipeline import create_workflow

# Créer le workflow
app = create_workflow()

# Poser une question
result = app.invoke({
    "question": "Qu'est-ce qu'une intégrale de Riemann ?",
    "chat_history": []
})

print(result["final_answer"])
```

### Mode Local avec Ollama

1. Installer Ollama : https://ollama.ai
2. Télécharger un modèle :

```bash
ollama pull mistral
# ou
ollama pull llama3
```

3. Configurer `.env` :

```bash
LLM_MODE=local
```

### Mode Cloud avec OpenAI

Configurer `.env` :

```bash
LLM_MODE=cloud
OPENAI_API_KEY=your_api_key_here
```

## Structure du Projet

```
math-rag-system/
├── config/                    # Fichiers de configuration
│   ├── config.yaml           # Configuration système
│   └── logging_config.yaml   # Configuration logs
│
├── data/                     # Données et caches
│   ├── raw/                  # PDFs sources
│   ├── processed/            # Textes extraits
│   ├── vector_store/         # Index FAISS
│   └── logs/                 # Logs d'exécution
│
├── docs/                     # Documentation
│   ├── PROJECT_STRUCTURE.md  # Structure du projet
│   └── TECHNICAL_DOCUMENTATION.md  # Documentation technique
│
├── scripts/                  # Scripts utilitaires
│   ├── build_vector_store.py
│   ├── download_pdfs.py
│   └── setup_gdrive.py
│
├── src/                      # Code source principal
│   ├── agents/               # Agents du workflow
│   │   ├── classifier.py
│   │   ├── planner.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── ...
│   │
│   ├── extraction/           # Extraction PDF
│   ├── interface/            # Interface Streamlit
│   ├── llm/                  # Clients LLM
│   ├── utils/                # Utilitaires
│   ├── vectorization/        # Système vectoriel
│   └── workflow/             # Workflow LangGraph
│
└── tests/                    # Tests
```

Voir [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) pour plus de détails.

## Tests

### Tests Unitaires

```bash
# Test rapide du workflow
python tests/test_quick.py

# Test complet avec tracing Langfuse
python tests/test_manual_tracing.py

# Tests de validation
python tests/test_validation.py
```

### Tests d'Intégration

```bash
# Test du flux complet
python tests/test_complete_flow.py

# Test des questions prédéfinies
python tests/run_test_questions.py
```

## Monitoring avec Langfuse

Si Langfuse est configuré, accédez au dashboard :

1. Ouvrir https://cloud.langfuse.com (ou votre instance self-hosted)
2. Visualiser les traces, spans, et métriques
3. Analyser les performances et coûts

Le système trace automatiquement :
- Chaque appel LLM avec input/output
- Latences et token usage
- Hiérarchie des appels (workflow → agents → LLM)
- Scores de qualité et métriques custom

## Documentation

- [README.md](README.md) - Ce fichier
- [QUICKSTART.md](QUICKSTART.md) - Guide de démarrage rapide
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Structure du projet
- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) - Documentation technique détaillée
- [TESTING.md](TESTING.md) - Guide des tests

## Développement

### Installation des Dépendances de Développement

```bash
pip install -r requirements-dev.txt
```

### Linting et Formatage

```bash
# Black pour formatage
black src/ tests/

# Flake8 pour linting
flake8 src/ tests/

# MyPy pour vérification de types
mypy src/
```

### Visualiser le Graphe LangGraph

```bash
python scripts/utils/visualize_graph.py
```

### Nettoyer les Caches

```bash
# Nettoyer les caches Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Nettoyer les logs
rm -rf data/logs/*.log

# Reconstruire le vector store
python scripts/build_vector_store.py --force
```

## Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Guidelines

- Suivre le style de code existant
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation si nécessaire
- S'assurer que tous les tests passent avant de soumettre

## Signaler un Bug

Ouvrir une issue sur GitHub avec :
- Description claire du problème
- Étapes pour reproduire
- Comportement attendu vs observé
- Environnement (OS, version Python, etc.)
- Logs pertinents

## Changelog

### Version 1.0.0 (2024)

- Workflow multi-agent avec LangGraph
- Recherche vectorielle FAISS
- Support Ollama (local) et OpenAI (cloud)
- Interface Streamlit avec LaTeX
- Observabilité Langfuse avec spans manuels
- Système de suggestions intelligentes
- Recherche web intégrée
- Tests unitaires et d'intégration

## License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Remerciements

- [LangGraph](https://github.com/langchain-ai/langgraph) pour l'orchestration multi-agent
- [FAISS](https://github.com/facebookresearch/faiss) pour la recherche vectorielle
- [Sentence Transformers](https://www.sbert.net/) pour les embeddings
- [Streamlit](https://streamlit.io/) pour l'interface web
- [Langfuse](https://langfuse.com/) pour l'observabilité LLM
- [Ollama](https://ollama.ai/) pour les modèles locaux

## Contact

Paul MONTIER - [@PaulMONTIER](https://github.com/PaulMONTIER)

Lien du projet : [https://github.com/PaulMONTIER/math-rag-system](https://github.com/PaulMONTIER/math-rag-system)

---

Si ce projet vous est utile, n'hésitez pas à lui donner une étoile !
