"""
Intégration Langfuse pour observabilité LLM.

Ce module fournit:
- Configuration Langfuse avec clés d'environnement
- Callback handler pour LangChain/LangGraph
- Décorateurs pour tracer les fonctions
- Utilitaires pour logging d'événements

Usage:
    from src.utils.langfuse_integration import get_langfuse_handler, trace_agent

    # Dans workflow
    handler = get_langfuse_handler()
    workflow.invoke({"question": "..."}, config={"callbacks": [handler]})

    # Décorateur pour agents
    @trace_agent("retriever")
    def retrieve_documents(query):
        ...
"""

import os
from typing import Optional, Dict, Any, List
from functools import wraps
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Langfuse
# ═══════════════════════════════════════════════════════════════════════════════

def is_langfuse_enabled() -> bool:
    """
    Vérifie si Langfuse est configuré et activé.

    Returns:
        True si les clés API sont présentes
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    return bool(public_key and secret_key)


def get_langfuse_client():
    """
    Retourne un client Langfuse configuré.

    Returns:
        Langfuse client ou None si désactivé
    """
    if not is_langfuse_enabled():
        logger.warning("Langfuse credentials not found in environment")
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        )

        logger.info("Langfuse client initialized successfully")
        return client

    except ImportError as e:
        logger.error(f"langfuse package not installed: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_langfuse_handler():
    """
    Retourne un CallbackHandler Langfuse pour LangChain/LangGraph.

    NOTE: Actuellement désactivé car incompatible avec LangChain 1.0+
    Utiliser get_langfuse_tracer() à la place pour tracer manuellement.

    Returns:
        None (désactivé)

    Example:
        >>> handler = get_langfuse_handler()
        >>> # Retourne None - utiliser get_langfuse_tracer() à la place
    """
    # Désactivé temporairement - incompatibilité LangChain 1.0
    # Le CallbackHandler Langfuse utilise l'ancienne API LangChain 0.x
    logger.warning("Langfuse CallbackHandler désactivé (incompatible LangChain 1.0+)")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Décorateurs de tracing
# ═══════════════════════════════════════════════════════════════════════════════

def trace_agent(agent_name: str):
    """
    Décorateur pour tracer l'exécution d'un agent.

    Args:
        agent_name: Nom de l'agent (classifier, retriever, etc.)

    Example:
        >>> @trace_agent("retriever")
        ... def retrieve_docs(query):
        ...     return search(query)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_langfuse_client()

            if client is None:
                # Langfuse désactivé, exécution normale
                return func(*args, **kwargs)

            try:
                # Créer une trace
                trace = client.trace(
                    name=f"agent_{agent_name}",
                    metadata={
                        "agent": agent_name,
                        "function": func.__name__
                    }
                )

                # Créer un span pour l'exécution
                span = trace.span(
                    name=func.__name__,
                    metadata={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                )

                # Exécuter fonction
                result = func(*args, **kwargs)

                # Logger le résultat
                span.end(output=str(result)[:500])

                return result

            except Exception as e:
                logger.error(f"Langfuse tracing error for {agent_name}: {e}")
                # Continuer l'exécution même si tracing échoue
                return func(*args, **kwargs)

        return wrapper
    return decorator


@contextmanager
def trace_workflow(workflow_name: str, metadata: Optional[Dict] = None):
    """
    Context manager pour tracer un workflow complet.

    Args:
        workflow_name: Nom du workflow
        metadata: Métadonnées additionnelles

    Example:
        >>> with trace_workflow("math_rag_query", {"user_id": "123"}):
        ...     result = app.invoke(state)
    """
    client = get_langfuse_client()

    if client is None:
        yield None
        return

    try:
        trace = client.trace(
            name=workflow_name,
            metadata=metadata or {}
        )

        yield trace

        trace.update(status="SUCCESS")

    except Exception as e:
        logger.error(f"Workflow tracing error: {e}")
        if trace:
            trace.update(status="ERROR", metadata={"error": str(e)})
        yield None


def log_agent_event(
    agent_name: str,
    event_type: str,
    data: Dict[str, Any],
    trace_id: Optional[str] = None
):
    """
    Log un événement d'agent dans Langfuse.

    Args:
        agent_name: Nom de l'agent
        event_type: Type d'événement (input, output, error, etc.)
        data: Données de l'événement
        trace_id: ID de trace parent (optionnel)

    Example:
        >>> log_agent_event(
        ...     "retriever",
        ...     "retrieval_complete",
        ...     {"docs_found": 5, "query": "intégrales"}
        ... )
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.score(
            name=f"{agent_name}_{event_type}",
            value=1.0,
            data_type="NUMERIC",
            comment=str(data),
            trace_id=trace_id
        )
    except Exception as e:
        logger.error(f"Failed to log agent event: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Métriques et scores
# ═══════════════════════════════════════════════════════════════════════════════

def log_quality_score(
    trace_id: str,
    score_name: str,
    value: float,
    comment: Optional[str] = None
):
    """
    Log un score de qualité dans Langfuse.

    Args:
        trace_id: ID de la trace
        score_name: Nom du score (confidence, relevance, etc.)
        value: Valeur du score (0.0 à 1.0)
        comment: Commentaire additionnel

    Example:
        >>> log_quality_score(
        ...     trace_id="abc123",
        ...     score_name="retrieval_confidence",
        ...     value=0.87,
        ...     comment="High confidence retrieval"
        ... )
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.score(
            trace_id=trace_id,
            name=score_name,
            value=value,
            data_type="NUMERIC",
            comment=comment
        )
        logger.debug(f"Logged quality score: {score_name}={value:.2f}")
    except Exception as e:
        logger.error(f"Failed to log quality score: {e}")


def log_user_feedback(
    trace_id: str,
    feedback: str,
    rating: Optional[int] = None
):
    """
    Log le feedback utilisateur dans Langfuse.

    Args:
        trace_id: ID de la trace
        feedback: Texte du feedback
        rating: Note optionnelle (1-5)

    Example:
        >>> log_user_feedback(
        ...     trace_id="abc123",
        ...     feedback="Réponse très utile!",
        ...     rating=5
        ... )
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        if rating:
            client.score(
                trace_id=trace_id,
                name="user_rating",
                value=rating,
                data_type="NUMERIC",
                comment=feedback
            )
        else:
            client.score(
                trace_id=trace_id,
                name="user_feedback",
                value=1.0,
                data_type="NUMERIC",
                comment=feedback
            )
        logger.info(f"User feedback logged for trace {trace_id}")
    except Exception as e:
        logger.error(f"Failed to log user feedback: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers pour LangGraph
# ═══════════════════════════════════════════════════════════════════════════════

def get_langfuse_tracer(trace_name: str, metadata: Optional[Dict] = None):
    """
    Crée un tracer Langfuse pour un workflow LangGraph.

    Utilise l'API directe Langfuse (sans CallbackHandler LangChain).
    Compatible avec LangChain 1.0+.

    Args:
        trace_name: Nom de la trace (ex: "math_rag_query")
        metadata: Métadonnées additionnelles

    Returns:
        Trace Langfuse ou None si désactivé

    Example:
        >>> trace = get_langfuse_tracer("rag_query", {"user": "paul"})
        >>> if trace:
        ...     # Log events manuellement
        ...     trace.span(name="retrieval", input={"query": "..."})
    """
    if not is_langfuse_enabled():
        return None

    client = get_langfuse_client()
    if not client:
        return None

    try:
        trace = client.trace(
            name=trace_name,
            metadata=metadata or {}
        )
        logger.debug(f"Langfuse trace created: {trace_name}")
        return trace
    except Exception as e:
        logger.error(f"Failed to create Langfuse trace: {e}")
        return None


def create_langgraph_config_with_langfuse(
    additional_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Crée une config LangGraph avec Langfuse activé.

    NOTE: Actuellement retourne config sans callbacks car incompatible LangChain 1.0
    Utiliser get_langfuse_tracer() pour tracer manuellement.

    Args:
        additional_config: Config additionnelle à fusionner

    Returns:
        Config dict (sans callbacks Langfuse)

    Example:
        >>> config = create_langgraph_config_with_langfuse({"recursion_limit": 25})
        >>> app.invoke(state, config=config)
    """
    config = additional_config or {}

    # CallbackHandler désactivé - incompatibilité LangChain 1.0
    logger.debug("Langfuse callbacks désactivés (utiliser get_langfuse_tracer)")

    return config


def flush_langfuse():
    """
    Force le flush des événements Langfuse vers le serveur.

    À appeler en fin de session ou avant fermeture.

    Example:
        >>> # En fin de requête
        >>> flush_langfuse()
    """
    try:
        from langfuse import Langfuse

        if is_langfuse_enabled():
            # Langfuse flush automatique, mais on peut forcer
            client = get_langfuse_client()
            if client:
                client.flush()
                logger.debug("Langfuse events flushed")
    except Exception as e:
        logger.error(f"Failed to flush Langfuse: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE LANGFUSE:
# - Traces: Workflow complet (1 par question utilisateur)
# - Spans: Opérations individuelles (retrieval, generation, etc.)
# - Scores: Métriques de qualité, feedback utilisateur
# - Events: Événements custom
#
# INTÉGRATION LANGGRAPH:
# 1. Utiliser CallbackHandler dans invoke() config
# 2. Chaque nœud du graphe → 1 span automatique
# 3. Appels LLM → tracés automatiquement
# 4. Métriques custom → log_quality_score()
#
# CONFIGURATION ENVIRONNEMENT:
# - LANGFUSE_PUBLIC_KEY: Clé publique (project)
# - LANGFUSE_SECRET_KEY: Clé secrète (API)
# - LANGFUSE_BASE_URL: URL du serveur (cloud ou self-hosted)
#
# FALLBACK GRACEFUL:
# - Si Langfuse non configuré → désactivation silencieuse
# - Pas d'interruption du workflow
# - Logs debug pour traçabilité
#
# DASHBOARD LANGFUSE:
# - URL: https://cloud.langfuse.com (ou self-hosted)
# - Visualisation: traces, spans, latences, coûts
# - Filtres: par agent, par user, par date
# - Métriques: token usage, latency p50/p95/p99
#
# OPTIMISATION:
# - Flush automatique toutes les 60s
# - Batch events pour performance
# - Pas de blocking I/O sur critical path
#
# SÉCURITÉ:
# - Ne jamais logger de données sensibles (PII)
# - Clés API en variables d'environnement uniquement
# - Option de désactivation en production si nécessaire
#
# ═══════════════════════════════════════════════════════════════════════════════
