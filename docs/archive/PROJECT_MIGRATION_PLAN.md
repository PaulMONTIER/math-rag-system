# Plan de Migration - Math RAG System vers Project A Requirements

## Date: 2025-11-17

## Objectif

Adapter le Math RAG System pour qu'il corresponde **parfaitement** aux exigences du Project A — Multi-Agent Research & Briefing Assistant.

---

## Architecture Actuelle vs. Requise

### Agents Actuels
1. ✅ **Classifier** - Route les questions
2. ✅ **Retriever** - Recherche dans ChromaDB
3. ✅ **Generator** - Génère les réponses
4. ✅ **Verifier** - Vérifie la qualité
5. ✅ **Suggester** (implicite dans generator) - Propose des questions

### Agents à Ajouter
6. ❌ **Web Search Agent** - Recherche externe (Tavily/DuckDuckGo)
7. ❌ **Editor/Critic Agent** - Review et édition
8. ❌ **Human Approval Agent** - Point d'interruption humain

---

## Exigences Manquantes à Implémenter

### 1. External Search Tool Agent

**Status**: ❌ Absent

**Ce qu'il faut faire**:
- Créer `src/agents/web_searcher.py`
- Intégrer API de recherche (DuckDuckGo - gratuite, pas de clé API)
- Alternative: Tavily (nécessite clé API mais meilleure qualité)
- L'agent doit :
  - Rechercher sur le web quand RAG local insuffisant
  - Extraire et résumer les résultats
  - Citer les sources web

**Fichiers à créer**:
- `src/agents/web_searcher.py`

**Fichiers à modifier**:
- `src/workflow/langgraph_pipeline.py` - ajouter noeud web_search
- `config/config.yaml` - configuration API

**Dépendances à ajouter**:
```python
# requirements.txt
duckduckgo-search>=4.0.0  # OU tavily-python>=0.2.0
```

---

### 2. SqliteSaver Persistence

**Status**: ❌ Absent

**Ce qu'il faut faire**:
- Utiliser `SqliteSaver` de LangGraph pour persistence
- Sauvegarder l'état du workflow à chaque étape
- Permettre reprise après interruption
- Stocker historique des approbations

**Fichiers à créer**:
- `data/checkpoints/` - dossier pour la DB SQLite

**Fichiers à modifier**:
- `src/workflow/langgraph_pipeline.py` - configurer SqliteSaver
- `src/interface/app.py` - gérer thread_id et reprises

**Code clé**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Dans create_rag_workflow
checkpointer = SqliteSaver.from_conn_string("data/checkpoints/workflow.db")
workflow = workflow_graph.compile(checkpointer=checkpointer)
```

---

### 3. Human-in-the-Loop Approval

**Status**: ❌ Absent

**Ce qu'il faut faire**:
- Créer un noeud d'interruption dans le workflow
- Interface Streamlit pour approval/edit/reject
- Sauvegarder les décisions humaines
- Permettre édition de la réponse avant validation

**Fichiers à créer**:
- `src/agents/human_approval.py`
- `src/interface/components/approval_interface.py`

**Fichiers à modifier**:
- `src/workflow/langgraph_pipeline.py` - ajouter interrupt_before
- `src/interface/app.py` - UI d'approbation

**Code clé**:
```python
# Dans le workflow
workflow = workflow_graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"]  # Interrupt avant approbation
)

# Dans l'interface
if state_needs_approval:
    # Afficher UI d'approbation
    action = show_approval_interface(draft_response)
    if action == "approve":
        continue_workflow()
    elif action == "edit":
        update_and_continue(edited_text)
    elif action == "reject":
        restart_workflow()
```

---

### 4. Langfuse Monitoring

**Status**: ⚠️ Configuré mais pas activé

**Ce qu'il faut faire**:
- Créer compte Langfuse (cloud ou self-hosted)
- Obtenir clés API (PUBLIC_KEY, SECRET_KEY)
- Configurer dans `.env`
- Ajouter décorateurs/callbacks LangGraph
- Créer dashboard avec métriques

**Fichiers à créer**:
- Aucun (configuration uniquement)

**Fichiers à modifier**:
- `.env` - ajouter clés Langfuse
- `src/workflow/langgraph_pipeline.py` - activer tracing
- `src/llm/closed_models.py` - callbacks Langfuse

**Configuration requise**:
```python
# .env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Dans le code
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler()
```

---

### 5. Restructuration du Workflow

**Status**: ❌ Workflow actuel trop simple

**Architecture Actuelle**:
```
Question → Classifier → Retriever → Generator → Response
```

**Architecture Requise (Project A)**:
```
Question
  → Planner (décide: RAG local OU web search OU les deux)
  → Retrieval Phase:
      ├─ Vector DB Agent (ChromaDB)
      └─ Web Search Agent (DuckDuckGo/Tavily)
  → Writer Agent (draft initial)
  → Critic/Editor Agent (review)
  → HUMAN APPROVAL (interrupt)
  → Final Response
```

**Fichiers à modifier**:
- `src/workflow/langgraph_pipeline.py` - restructurer graphe complet

---

## Plan d'Implémentation (Ordre Recommandé)

### Phase 1: Persistence (Fondation)
**Durée estimée**: 30 min

1. Installer SqliteSaver
2. Modifier create_rag_workflow pour utiliser checkpointer
3. Tester sauvegarde/reprise basique

### Phase 2: Web Search Agent
**Durée estimée**: 1h

1. Installer duckduckgo-search
2. Créer WebSearchAgent
3. Intégrer au workflow
4. Tester recherches web

### Phase 3: Restructuration Workflow
**Durée estimée**: 1h30

1. Créer Editor/Critic Agent
2. Restructurer le graphe LangGraph
3. Ajouter routing intelligent (local vs web)
4. Tests du nouveau flow

### Phase 4: Human-in-the-Loop
**Durée estimée**: 2h

1. Créer human_approval node
2. Interface Streamlit d'approbation
3. Gestion approve/edit/reject
4. Integration avec persistence

### Phase 5: Langfuse Monitoring
**Durée estimée**: 45 min

1. Créer compte Langfuse
2. Configurer clés API
3. Activer tracing
4. Créer dashboard

### Phase 6: Documentation & Tests
**Durée estimée**: 1h

1. Architecture diagram
2. Rapport technique
3. Screenshots Langfuse
4. Tests end-to-end

**DURÉE TOTALE ESTIMÉE**: ~7h de travail

---

## Nouveau Workflow Détaillé

```python
# Nouveau graphe LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def create_research_workflow(config):
    # State
    class ResearchState(TypedDict):
        question: str
        question_type: str
        search_strategy: str  # "local_only", "web_only", "both"
        local_docs: list
        web_results: list
        draft_response: str
        edited_response: str
        human_feedback: str
        final_response: str
        sources: list
        approval_status: str  # "pending", "approved", "rejected"

    # Graph
    workflow = StateGraph(ResearchState)

    # Nodes
    workflow.add_node("planner", plan_search_strategy)
    workflow.add_node("vector_retrieval", retrieve_from_db)
    workflow.add_node("web_search", search_web)
    workflow.add_node("writer", generate_draft)
    workflow.add_node("critic", review_and_edit)
    workflow.add_node("human_approval", wait_for_human)

    # Edges (routing)
    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_search,
        {
            "local": "vector_retrieval",
            "web": "web_search",
            "both": "vector_retrieval"
        }
    )

    workflow.add_edge("vector_retrieval", "writer")
    workflow.add_edge("web_search", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "human_approval")

    workflow.add_conditional_edges(
        "human_approval",
        check_approval,
        {
            "approved": END,
            "edit": "writer",
            "reject": "planner"
        }
    )

    # Compile with persistence and interrupts
    checkpointer = SqliteSaver.from_conn_string("data/checkpoints/research.db")

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"]
    )
```

---

## Checklist de Validation Finale

Avant de soumettre le projet, vérifier que :

### Multi-Agent Architecture
- [ ] Au moins 5 agents distincts avec rôles clairs
- [ ] Routing dynamique entre agents
- [ ] Chaque agent a une responsabilité unique

### Vector Database
- [ ] ChromaDB fonctionnel
- [ ] Retrieval avec scoring
- [ ] Citations des sources locales

### External Search Tool
- [ ] Agent de recherche web implémenté
- [ ] Intégration DuckDuckGo ou Tavily
- [ ] Citations des sources web

### Human-in-the-Loop
- [ ] Point d'interruption dans workflow
- [ ] Interface approve/edit/reject
- [ ] Reprise après validation
- [ ] Historique des décisions sauvegardé

### Persistence
- [ ] SqliteSaver configuré
- [ ] État sauvegardé à chaque étape
- [ ] Reprise possible après crash
- [ ] Thread management fonctionnel

### Langfuse Monitoring
- [ ] Compte Langfuse créé
- [ ] Clés API configurées
- [ ] Traces visibles dans dashboard
- [ ] Spans pour chaque agent
- [ ] Screenshot du dashboard

### Documentation
- [ ] Architecture diagram créé
- [ ] Rapport technique rédigé
- [ ] Flow example documenté
- [ ] README à jour

### Demo
- [ ] Application fonctionnelle
- [ ] Tous les modes testés
- [ ] Exemples de questions préparés
- [ ] Scénario de démo répété

---

## Dépendances Supplémentaires

Ajouter à `requirements.txt`:

```txt
# Web Search
duckduckgo-search>=4.0.0

# Persistence
langgraph>=0.0.40  # Mise à jour si nécessaire

# Monitoring
langfuse>=2.0.0

# Optional: meilleure recherche web
# tavily-python>=0.2.0
```

---

## Notes de Migration

### Compatibilité Backward
- L'ancien workflow doit rester fonctionnel
- Ajouter flag `use_classic_workflow` dans config
- Permettre switch entre ancien/nouveau mode

### Tests
- Tester chaque agent indépendamment
- Tests d'intégration du workflow complet
- Tests de persistence et reprise
- Tests d'approbation humaine

### Performance
- Web search peut être lent (5-10s)
- Ajouter caching des résultats web
- Timeout approprié pour chaque agent

---

**Status**: Plan créé, prêt pour implémentation
**Prochaine étape**: Phase 1 - Persistence
