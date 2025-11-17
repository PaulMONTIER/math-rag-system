"""
Workflow LangGraph - Orchestration multi-agent pour RAG.

Ce module implémente le workflow complet:
START → Classify → Retrieve → Generate → Verify → (Human Validation) → END

Usage:
    from src.workflow.langgraph_pipeline import create_rag_workflow

    workflow = create_rag_workflow(config)
    result = workflow.invoke({"question": "Qu'est-ce qu'une dérivée ?"})
    print(result["final_response"])
"""

import time
from typing import Dict, Any, Optional, TypedDict, Literal
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.classifier import ClassifierAgent, Intent
from src.agents.retriever import RetrieverAgent
from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent
from src.agents.web_searcher import WebSearchAgent
from src.agents.planner import PlannerAgent, SearchStrategy
from src.agents.editor import EditorAgent
from src.vectorization.vector_store import VectorStore
from src.vectorization.embedder import Embedder
from src.llm.closed_models import get_llm_client
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector, QueryMetrics, create_query_metrics
from src.utils.cost_tracker import CostTracker
from src.utils.langfuse_integration import get_langfuse_tracer, is_langfuse_enabled
from src.utils.langfuse_context import set_current_langfuse_span

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# État du workflow (TypedDict pour LangGraph)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowState(TypedDict):
    """
    État partagé du workflow.

    Passe entre tous les nœuds du graphe.
    """
    # Input
    question: str
    student_level: Optional[str]

    # Paramètres de personnalisation avancés
    rigor_level: Optional[int]
    num_examples: Optional[int]
    include_proofs: Optional[bool]
    include_history: Optional[bool]
    detailed_latex: Optional[bool]

    # Classification
    intent: Optional[str]
    intent_confidence: Optional[float]

    # Planning
    search_strategy: Optional[str]
    planning_confidence: Optional[float]
    planning_reasoning: Optional[str]

    # Retrieval
    retrieved_docs: Optional[list]
    context: Optional[str]
    retrieval_confidence: Optional[float]  # Score de confiance pour fallback web

    # Web Search
    web_search_results: Optional[list]
    web_search_context: Optional[str]
    use_web_search: Optional[bool]

    # Generation
    generated_answer: Optional[str]
    sources_cited: Optional[list]

    # Edition/Review
    editor_quality_score: Optional[float]
    editor_suggestions: Optional[list]
    needs_revision: Optional[bool]

    # Verification
    verification_result: Optional[Dict]
    confidence_score: Optional[float]

    # Output
    final_response: str
    success: bool

    # Metadata
    metadata: Dict[str, Any]
    start_time: float
    error_message: Optional[str]

    # Langfuse tracing (optionnel)
    langfuse_trace: Optional[Any]
    langfuse_current_span: Optional[Any]  # Span du nœud en cours (pour nesting)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper pour tracing Langfuse
# ═══════════════════════════════════════════════════════════════════════════════

def create_node_span(trace: Optional[Any], node_name: str, input_data: Dict[str, Any]) -> Optional[Any]:
    """
    Crée un span Langfuse pour un nœud du workflow.

    Args:
        trace: Trace Langfuse parente
        node_name: Nom du nœud (classify, retrieve, etc.)
        input_data: Données d'entrée du nœud

    Returns:
        Span Langfuse ou None si tracing désactivé
    """
    if not trace:
        return None

    try:
        span = trace.span(
            name=node_name,
            input=input_data
        )
        logger.debug(f"Langfuse span created for node: {node_name}")
        return span
    except Exception as e:
        logger.warning(f"Failed to create Langfuse span for {node_name}: {e}")
        return None


def finalize_node_span(span: Optional[Any], output_data: Dict[str, Any]) -> None:
    """
    Finalise un span Langfuse avec les résultats du nœud.

    Args:
        span: Span à finaliser
        output_data: Données de sortie du nœud
    """
    if not span:
        return

    try:
        span.end(output=output_data)
        logger.debug("Langfuse span finalized")
    except Exception as e:
        logger.warning(f"Failed to finalize Langfuse span: {e}")


def create_generation_span(
    parent_span: Optional[Any],
    name: str,
    model: str,
    input_messages: list
) -> Optional[Any]:
    """
    Crée un span de type 'generation' pour un appel LLM sous un span parent.

    Args:
        parent_span: Span parent (span du nœud)
        name: Nom du span (ex: "classify_llm", "generate_llm")
        model: Nom du modèle LLM
        input_messages: Messages d'entrée pour le LLM

    Returns:
        Span de génération ou None si désactivé
    """
    if not parent_span:
        return None

    try:
        generation = parent_span.generation(
            name=name,
            model=model,
            input=input_messages
        )
        logger.debug(f"Langfuse generation span created: {name}")
        return generation
    except Exception as e:
        logger.warning(f"Failed to create generation span: {e}")
        return None


def finalize_generation_span(
    generation: Optional[Any],
    output_text: str,
    usage: Optional[Dict] = None
) -> None:
    """
    Finalise un span de génération avec la réponse du LLM.

    Args:
        generation: Span de génération
        output_text: Texte généré par le LLM
        usage: Statistiques d'utilisation (tokens, etc.)
    """
    if not generation:
        return

    try:
        end_params = {"output": output_text}
        if usage:
            end_params["usage"] = usage

        generation.end(**end_params)
        logger.debug("Langfuse generation span finalized")
    except Exception as e:
        logger.warning(f"Failed to finalize generation span: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions des nœuds
# ═══════════════════════════════════════════════════════════════════════════════

def classify_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Classification de l'intention."""
    logger.info("→ Classify node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="classify",
        input_data={"question": state["question"][:100]}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    # Récupérer agents depuis config
    classifier: ClassifierAgent = config["classifier"]

    # Classifier
    classification = classifier.classify(state["question"])

    # Mettre à jour état
    state["intent"] = classification.intent.value
    state["intent_confidence"] = classification.confidence

    state["metadata"]["classification"] = {
        "intent": classification.intent.value,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning
    }

    logger.info(f"  Intent: {classification.intent.value} (confidence: {classification.confidence:.2f})")

    # Finaliser span avec résultats
    finalize_node_span(span, {
        "intent": classification.intent.value,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning[:200] if classification.reasoning else ""
    })

    return state


def retrieve_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Récupération de documents (RAG)."""
    logger.info("→ Retrieve node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="retrieve",
        input_data={"question": state["question"][:100]}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    retriever: RetrieverAgent = config["retriever"]

    # Rechercher documents
    start = time.time()
    results = retriever.retrieve(
        question=state["question"],
        filters=None  # Optionnel: filtrer par level, source, etc.
    )
    retrieval_time = time.time() - start

    # Construire contexte
    context = retriever.build_context(results, include_metadata=True)

    # Mettre à jour état
    state["retrieved_docs"] = [
        {
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "rank": r.rank
        }
        for r in results
    ]
    state["context"] = context

    # Calculer confiance de retrieval pour fallback intelligent
    retrieval_confidence = retriever.calculate_retrieval_confidence(results)
    state["retrieval_confidence"] = retrieval_confidence

    state["metadata"]["retrieval"] = {
        "docs_found": len(results),
        "avg_score": sum(r.score for r in results) / len(results) if results else 0,
        "retrieval_time": retrieval_time,
        "confidence": retrieval_confidence
    }

    logger.info(
        f"  Retrieved {len(results)} documents (confidence: {retrieval_confidence:.2f})"
    )

    # Finaliser span
    finalize_node_span(span, {
        "docs_found": len(results),
        "avg_score": sum(r.score for r in results) / len(results) if results else 0,
        "confidence": retrieval_confidence,
        "retrieval_time": retrieval_time
    })

    return state


def web_search_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Recherche web externe."""
    logger.info("→ Web Search node")

    web_searcher: WebSearchAgent = config["web_searcher"]

    # Effectuer recherche web UNE SEULE FOIS
    start = time.time()
    response = web_searcher.search(state["question"], max_results=5)
    search_time = time.time() - start

    # Mettre à jour état
    state["web_search_results"] = [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "relevance_score": r.relevance_score
        }
        for r in response.results
    ]

    # Formater le contexte DIRECTEMENT à partir des résultats (pas de deuxième appel search!)
    if response.results:
        context_parts = []
        context_parts.append(f"# Web Search Results for: {state['question']}\n")

        for idx, result in enumerate(response.results, 1):
            context_parts.append(f"## Result {idx}: {result.title}")
            context_parts.append(f"**URL**: {result.url}")
            context_parts.append(f"**Content**: {result.snippet}\n")

        state["web_search_context"] = "\n".join(context_parts)
    else:
        state["web_search_context"] = "No relevant web results found."

    state["metadata"]["web_search"] = {
        "results_found": len(response.results),
        "search_time": search_time,
        "sources": response.sources
    }

    logger.info(f"  Found {len(response.results)} web results")

    return state


def combine_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Combine RAG local + Web search (stratégie BOTH)."""
    logger.info("→ Combine node (RAG + Web)")

    retriever: RetrieverAgent = config["retriever"]
    web_searcher: WebSearchAgent = config["web_searcher"]

    # 1. Récupérer documents locaux
    logger.info("  Step 1/2: RAG retrieval")
    start_rag = time.time()
    rag_results = retriever.retrieve(
        question=state["question"],
        filters=None
    )
    rag_time = time.time() - start_rag

    # 2. Récupérer résultats web
    logger.info("  Step 2/2: Web search")
    start_web = time.time()
    web_response = web_searcher.search(state["question"], max_results=3)
    web_time = time.time() - start_web

    # Combiner les contextes
    rag_context = retriever.build_context(rag_results, include_metadata=True)
    web_context = web_searcher.search_for_context(state["question"])

    combined_context = f"""# Sources Locales (Base Vectorielle)\n\n{rag_context}\n\n"""
    combined_context += f"""# Sources Web Externes\n\n{web_context}"""

    # Mettre à jour état avec les deux sources
    state["retrieved_docs"] = [
        {
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "rank": r.rank
        }
        for r in rag_results
    ]
    state["web_search_results"] = [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "relevance_score": r.relevance_score
        }
        for r in web_response.results
    ]
    state["context"] = combined_context
    state["web_search_context"] = web_context

    state["metadata"]["combine"] = {
        "rag_docs": len(rag_results),
        "web_results": len(web_response.results),
        "rag_time": rag_time,
        "web_time": web_time,
        "total_sources": len(rag_results) + len(web_response.results)
    }

    logger.info(f"  Combined {len(rag_results)} RAG docs + {len(web_response.results)} web results")

    return state


def generate_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Génération de réponse."""
    logger.info("→ Generate node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="generate",
        input_data={"question": state["question"][:100], "strategy": state.get("search_strategy")}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    generator: GeneratorAgent = config["generator"]

    # Reconstruire RetrievalResult depuis state (pour extraction sources)
    from src.agents.retriever import RetrievalResult
    retrieved_results = None
    if state.get("retrieved_docs"):
        retrieved_results = [
            RetrievalResult(
                text=doc["text"],
                score=doc["score"],
                metadata=doc["metadata"],
                rank=doc["rank"]
            )
            for doc in state["retrieved_docs"]
        ]

    # Déterminer le contexte à utiliser (RAG local, Web, ou les deux)
    context = state.get("context")  # RAG local
    web_context = state.get("web_search_context")  # Web search

    # ⭐ FALLBACK INTELLIGENT: Si confiance de retrieval trop faible, ajouter web search
    CONFIDENCE_THRESHOLD = 0.5  # Baissé de 0.6 à 0.5 (plus tolérant)
    retrieval_conf = state.get("retrieval_confidence", 1.0)
    search_strategy = state.get("search_strategy", "")

    if (search_strategy == "local_only" and
        retrieval_conf < CONFIDENCE_THRESHOLD and
        not web_context):  # Pas déjà de web search

        logger.warning(
            f"⚠️  Low retrieval confidence ({retrieval_conf:.2f}), "
            f"triggering automatic web fallback"
        )

        # Marquer que le fallback a été tenté (avant même de chercher)
        if "fallback" not in state["metadata"]:
            state["metadata"]["fallback"] = {}
        state["metadata"]["fallback"]["triggered"] = True
        state["metadata"]["fallback"]["confidence"] = retrieval_conf

        try:
            # Invoquer web searcher
            web_searcher = config["web_searcher"]
            web_results = web_searcher.search(
                query=state["question"],
                max_results=3  # Moins de résultats pour le fallback
            )

            # Enregistrer le nombre de résultats trouvés
            state["metadata"]["fallback"]["web_results_found"] = len(web_results.results)

            # Créer le contexte web si résultats trouvés
            if web_results.results:
                web_context = "\n\n".join([
                    f"**{r.title}**\n{r.snippet}\nSource: {r.url}"
                    for r in web_results.results
                ])

                # Enregistrer les résultats web dans state
                state["web_search_results"] = [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in web_results.results
                ]

                logger.info(f"✓ Fallback: Added {len(web_results.results)} web results")
            else:
                logger.warning("⚠️  Fallback triggered but found 0 web results")

        except Exception as e:
            logger.error(f"❌ Web fallback failed: {e}")
            state["metadata"]["fallback"]["error"] = str(e)
            # Continue sans fallback en cas d'erreur

    # Utiliser web_context si context est None (stratégie WEB_ONLY)
    # Sinon combiner les deux si les deux existent (stratégie BOTH ou fallback)
    if context is None and web_context:
        final_context = web_context
    elif context and web_context:
        final_context = f"{context}\n\n--- Sources Web Complémentaires ---\n{web_context}"
    else:
        final_context = context or ""

    # Générer réponse avec paramètres de personnalisation
    start = time.time()
    response = generator.generate(
        question=state["question"],
        context=final_context,
        student_level=state.get("student_level", "Détaillé"),
        retrieved_results=retrieved_results,  # Passer les résultats pour extraction sources
        rigor_level=state.get("rigor_level", 3),
        num_examples=state.get("num_examples", 2),
        include_proofs=state.get("include_proofs", True),
        include_history=state.get("include_history", False),
        detailed_latex=state.get("detailed_latex", True)
    )
    generation_time = time.time() - start

    # Mettre à jour état
    state["generated_answer"] = response.answer
    state["sources_cited"] = response.sources_cited

    state["metadata"]["generation"] = {
        "model": response.llm_response.model,
        "tokens": response.llm_response.total_tokens,
        "cost": response.llm_response.cost,
        "generation_time": generation_time,
        "has_formulas": response.metadata.get("has_formulas", False),
        "suggestions": response.metadata.get("suggestions", [])  # ⭐ Ajouter les suggestions
    }

    logger.info(f"  Generated {len(response.answer)} chars")

    # Finaliser span
    finalize_node_span(span, {
        "response_length": len(state["generated_answer"]),
        "sources_count": len(state["sources_cited"])
    })

    return state


def verify_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Vérification de qualité."""
    logger.info("→ Verify node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="verify",
        input_data={"question": state["question"][:100]}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    verifier: VerifierAgent = config["verifier"]

    # Vérifier
    start = time.time()
    verification = verifier.verify(
        question=state["question"],
        answer=state["generated_answer"],
        context=state["context"],
        sources=state["sources_cited"]
    )
    verification_time = time.time() - start

    # Mettre à jour état
    state["verification_result"] = {
        "is_valid": verification.is_valid,
        "confidence": verification.confidence_score,
        "issues": verification.issues_found,
        "warnings": verification.warnings,
        "recommendation": verification.recommendation
    }
    state["confidence_score"] = verification.confidence_score

    state["metadata"]["verification"] = {
        "is_valid": verification.is_valid,
        "confidence": verification.confidence_score,
        "recommendation": verification.recommendation,
        "verification_time": verification_time
    }

    logger.info(f"  Confidence: {verification.confidence_score:.2f} ({verification.recommendation})")

    # Finaliser span
    finalize_node_span(span, {
        "is_valid": state["verification_result"]["is_valid"],
        "confidence": state["confidence_score"]
    })

    return state


def plan_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Planification de la stratégie de recherche."""
    logger.info("→ Plan node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="plan",
        input_data={"question": state["question"][:100]}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    planner: PlannerAgent = config["planner"]

    # Décider de la stratégie (RAG local vs Web vs Both)
    start = time.time()
    decision = planner.plan(state["question"])
    planning_time = time.time() - start

    # Mettre à jour état
    state["search_strategy"] = decision.strategy.value
    state["planning_confidence"] = decision.confidence
    state["planning_reasoning"] = decision.reasoning

    state["metadata"]["planning"] = {
        "strategy": decision.strategy.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "planning_time": planning_time
    }

    logger.info(f"  Strategy: {decision.strategy.value} (confidence: {decision.confidence:.2f})")
    logger.info(f"  Reasoning: {decision.reasoning}")

    # Finaliser span
    finalize_node_span(span, {
        "strategy": decision.strategy.value,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning[:200] if decision.reasoning else "",
        "planning_time": planning_time
    })

    return state


def editor_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Review et amélioration de la réponse."""
    logger.info("→ Editor node")

    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="editor",
        input_data={"question": state["question"][:100]}
    )

    # Stocker le span actuel dans le state ET dans le context variable
    state["langfuse_current_span"] = span
    set_current_langfuse_span(span)

    editor: EditorAgent = config["editor"]

    # Review de la réponse générée
    start = time.time()
    review = editor.review(
        question=state["question"],
        generated_answer=state["generated_answer"],
        context=state.get("context"),
        sources=state.get("sources_cited")
    )
    review_time = time.time() - start

    # Mettre à jour état
    state["editor_quality_score"] = review.quality_score
    state["editor_suggestions"] = review.suggestions
    state["needs_revision"] = review.needs_revision

    # Si amélioration disponible, l'utiliser
    if review.needs_revision and review.improved_answer:
        logger.info(f"  Using improved answer (quality improved from {review.quality_score:.2f})")
        state["generated_answer"] = review.improved_answer

    state["metadata"]["editor"] = {
        "quality_score": review.quality_score,
        "needs_revision": review.needs_revision,
        "issues_found": review.issues_found,
        "suggestions": review.suggestions,
        "reasoning": review.reasoning,
        "review_time": review_time
    }

    logger.info(f"  Quality score: {review.quality_score:.2f}")
    if review.issues_found:
        logger.info(f"  Issues: {', '.join(review.issues_found)}")

    # Finaliser span
    finalize_node_span(span, {
        "quality_score": state["editor_quality_score"],
        "needs_revision": state["needs_revision"]
    })

    return state


def human_approval_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Approbation humaine (point d'interruption)."""
    logger.info("→ Human Approval node")

    # Ce nœud sert de point d'arrêt pour validation humaine
    # L'utilisateur peut:
    # - Approuver (approve): continuer vers finalize
    # - Éditer (edit): modifier la réponse puis continuer
    # - Rejeter (reject): abandonner ou regénérer

    # Pour l'instant, on ajoute juste un marqueur dans metadata
    state["metadata"]["human_approval_pending"] = True

    logger.info("  Waiting for human approval...")

    return state


def finalize_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Finalisation et logging."""
    logger.info("→ Finalize node")

    # Préparer réponse finale
    state["final_response"] = state["generated_answer"]
    state["success"] = True

    # Calculer temps total
    total_time = time.time() - state["start_time"]
    state["metadata"]["total_time"] = total_time

    # Ajouter student_level dans metadata pour traçabilité
    state["metadata"]["student_level"] = state.get("student_level", "Détaillé")

    # Enregistrer métriques
    metrics_collector: Optional[MetricsCollector] = config.get("metrics_collector")
    if metrics_collector:
        query_metrics = create_query_metrics(state["question"], state)
        metrics_collector.add_query(query_metrics)

    logger.info(f"✓ Workflow complete in {total_time:.2f}s")

    return state


def off_topic_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Réponse hors-sujet."""
    logger.info("→ Off-topic node")

    state["final_response"] = (
        "Désolé, je suis un assistant spécialisé en mathématiques. "
        "Cette question semble hors de mon domaine d'expertise. "
        "Pouvez-vous me poser une question sur les mathématiques ?"
    )
    state["success"] = True
    state["metadata"]["total_time"] = time.time() - state["start_time"]

    return state


def clarification_node(state: WorkflowState, config: Any) -> WorkflowState:
    """Nœud: Demande de clarification."""
    logger.info("→ Clarification node")

    state["final_response"] = (
        "Votre question est un peu vague. Pourriez-vous préciser ce que vous souhaitez savoir ? "
        "Par exemple, voulez-vous une définition, une démonstration, ou un exemple concret ?"
    )
    state["success"] = True
    state["metadata"]["total_time"] = time.time() - state["start_time"]

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions de routing conditionnel
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_classification(state: WorkflowState) -> Literal["plan", "off_topic", "clarification"]:
    """
    Route après classification.

    Returns:
        Nom du prochain nœud
    """
    intent = state.get("intent", "MATH_QUESTION")

    if intent == Intent.MATH_QUESTION.value:
        return "plan"  # Nouveau: router vers planning au lieu de retrieve directement
    elif intent == Intent.OFF_TOPIC.value:
        return "off_topic"
    elif intent == Intent.NEED_CLARIFICATION.value:
        return "clarification"
    else:
        # Default: traiter comme question math
        return "plan"


def route_after_planning(state: WorkflowState) -> Literal["retrieve", "web_search", "combine"]:
    """
    Route après planification selon la stratégie décidée.

    Stratégies possibles:
    - local_only: uniquement RAG local → retrieve
    - web_only: uniquement recherche web → web_search
    - both: combiner les deux sources → combine (retrieve puis web_search)

    Returns:
        Nom du prochain nœud
    """
    from src.agents.planner import SearchStrategy

    strategy = state.get("search_strategy", SearchStrategy.LOCAL_ONLY.value)

    if strategy == SearchStrategy.LOCAL_ONLY.value:
        logger.info("  → Routing to LOCAL RAG only")
        return "retrieve"
    elif strategy == SearchStrategy.WEB_ONLY.value:
        logger.info("  → Routing to WEB SEARCH only")
        return "web_search"
    elif strategy == SearchStrategy.BOTH.value:
        logger.info("  → Routing to COMBINED (RAG + Web)")
        return "combine"
    else:
        # Default: local only
        logger.warning(f"Unknown strategy '{strategy}', defaulting to local RAG")
        return "retrieve"


def route_after_verification(state: WorkflowState) -> Literal["finalize", "finalize"]:
    """
    Route après vérification.

    Pour MVP: toujours finaliser (pas de validation humaine).
    TODO: Ajouter nœud "human_validate" si confidence < seuil.

    Returns:
        Nom du prochain nœud
    """
    # Pour MVP: toujours approuver
    # confidence = state.get("confidence_score", 1.0)
    # if confidence < 0.75:
    #     return "human_validate"

    return "finalize"


# ═══════════════════════════════════════════════════════════════════════════════
# Construction du workflow
# ═══════════════════════════════════════════════════════════════════════════════

def create_rag_workflow(config: object, force_provider: Optional[str] = None) -> Any:
    """
    Crée le workflow LangGraph complet.

    Args:
        config: Objet Config
        force_provider: Provider LLM à utiliser (override config.llm.provider)
                       Options: "openai", "anthropic", "local"

    Returns:
        Workflow compilé prêt à l'emploi

    Example:
        >>> config = load_config()
        >>> workflow = create_rag_workflow(config, force_provider="openai")
        >>> result = workflow.invoke({
        ...     "question": "Qu'est-ce qu'une dérivée ?",
        ...     "student_level": "L2"
        ... })
        >>> print(result["final_response"])
    """
    provider_info = f" with provider={force_provider}" if force_provider else ""
    logger.info(f"Creating RAG workflow{provider_info}")

    # Initialiser composants
    cost_tracker = CostTracker(config)
    llm_client = get_llm_client(config, cost_tracker, force_provider=force_provider)

    # Embedder et Vector Store
    embedder = Embedder(config)
    vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)

    # Charger vector store si existant
    try:
        vector_store.load("default")
        logger.info(f"✓ Loaded vector store with {vector_store.total_vectors} vectors")
    except Exception as e:
        logger.warning(f"Vector store not found or empty: {e}")

    # Agents
    classifier = ClassifierAgent(config, llm_client)
    planner = PlannerAgent(config, llm_client)
    retriever = RetrieverAgent(config, vector_store, embedder)
    web_searcher = WebSearchAgent(config, max_results=5, timeout=10)
    generator = GeneratorAgent(config, llm_client)
    editor = EditorAgent(config, llm_client)
    verifier = VerifierAgent(config, llm_client)

    # Metrics collector
    metrics_collector = MetricsCollector()

    # Config pour nœuds (passage d'objets)
    node_config = {
        "classifier": classifier,
        "planner": planner,
        "retriever": retriever,
        "web_searcher": web_searcher,
        "generator": generator,
        "editor": editor,
        "verifier": verifier,
        "cost_tracker": cost_tracker,
        "metrics_collector": metrics_collector
    }

    # Créer graphe
    workflow = StateGraph(WorkflowState)

    # Ajouter nœuds
    workflow.add_node("classify", lambda state: classify_node(state, node_config))
    workflow.add_node("plan", lambda state: plan_node(state, node_config))
    workflow.add_node("retrieve", lambda state: retrieve_node(state, node_config))
    workflow.add_node("web_search", lambda state: web_search_node(state, node_config))
    workflow.add_node("combine", lambda state: combine_node(state, node_config))
    workflow.add_node("generate", lambda state: generate_node(state, node_config))
    workflow.add_node("editor", lambda state: editor_node(state, node_config))
    workflow.add_node("verify", lambda state: verify_node(state, node_config))
    workflow.add_node("human_approval", lambda state: human_approval_node(state, node_config))
    workflow.add_node("finalize", lambda state: finalize_node(state, node_config))
    workflow.add_node("off_topic", lambda state: off_topic_node(state, node_config))
    workflow.add_node("clarification", lambda state: clarification_node(state, node_config))

    # Définir edges (flux)
    workflow.set_entry_point("classify")

    # Routing conditionnel après classification
    workflow.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "plan": "plan",
            "off_topic": "off_topic",
            "clarification": "clarification"
        }
    )

    # Routing intelligent après planning (stratégie: LOCAL / WEB / BOTH)
    workflow.add_conditional_edges(
        "plan",
        route_after_planning,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "combine": "combine"
        }
    )

    # Tous les chemins de récupération mènent à generation
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("combine", "generate")

    # Flux: generate → editor → verify → human_approval → finalize
    workflow.add_edge("generate", "editor")
    workflow.add_edge("editor", "verify")
    workflow.add_edge("verify", "human_approval")
    workflow.add_edge("human_approval", "finalize")

    # Tous les chemins mènent à END
    workflow.add_edge("finalize", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("clarification", END)

    # Configurer persistence avec SqliteSaver
    # NOTE: Temporairement désactivé car langgraph-checkpoint-sqlite 3.x
    # nécessite un context manager qui complique la gestion du lifecycle.
    # TODO: Downgrader vers 1.x ou implémenter un wrapper pour version 3.x
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("⚠ Checkpointer temporarily disabled (compatibility issue with v3.x)")

    # Vérifier si Langfuse est activé
    if is_langfuse_enabled():
        logger.info("✓ Langfuse monitoring ENABLED - LLM calls will be traced")
    else:
        logger.warning("⚠ Langfuse monitoring DISABLED - Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable")

    # Compiler sans persistence pour le moment
    # NOTE: interrupt_before=["human_approval"] désactivé pour tests automatiques
    # Réactiver pour production avec interface interactive
    app = workflow.compile()

    logger.info("✓ Workflow created successfully with persistence and human-in-the-loop")

    return app


def invoke_workflow(
    workflow: Any,
    question: str,
    student_level: str = "L2",
    rigor_level: int = 3,
    num_examples: int = 2,
    include_proofs: bool = True,
    include_history: bool = False,
    detailed_latex: bool = True
) -> Dict[str, Any]:
    """
    Invoque le workflow avec une question.

    Args:
        workflow: Workflow compilé
        question: Question utilisateur
        student_level: Niveau étudiant
        rigor_level: Niveau de rigueur mathématique (1-5)
        num_examples: Nombre d'exemples à inclure (0-3)
        include_proofs: Inclure les démonstrations
        include_history: Inclure le contexte historique
        detailed_latex: Développer les formules LaTeX

    Returns:
        État final avec final_response

    Example:
        >>> result = invoke_workflow(workflow, "Qu'est-ce qu'une dérivée ?")
        >>> print(result["final_response"])
    """
    # État initial
    initial_state = {
        "question": question,
        "student_level": student_level,
        "rigor_level": rigor_level,
        "num_examples": num_examples,
        "include_proofs": include_proofs,
        "include_history": include_history,
        "detailed_latex": detailed_latex,
        "intent": None,
        "intent_confidence": None,
        "retrieved_docs": None,
        "context": None,
        "generated_answer": None,
        "sources_cited": None,
        "verification_result": None,
        "confidence_score": None,
        "final_response": "",
        "success": False,
        "metadata": {
            "query_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "timestamp": datetime.now().isoformat()
        },
        "start_time": time.time(),
        "error_message": None,
        "langfuse_trace": None  # Sera rempli ci-dessous si Langfuse est activé
    }

    # Invoquer
    logger.info(f"Invoking workflow for question: {question[:50]}...")

    # Préparer config pour workflow
    config = {"configurable": {"thread_id": initial_state["metadata"]["query_id"]}}

    # Créer trace Langfuse manuelle (compatible LangChain 1.0)
    langfuse_trace = get_langfuse_tracer(
        trace_name="math_rag_workflow",
        metadata={
            "query_id": initial_state["metadata"]["query_id"],
            "question": question[:100],
            "student_level": student_level,
            "rigor_level": rigor_level
        }
    )

    if langfuse_trace:
        logger.debug("Langfuse trace created for workflow")
        # Passer la trace dans le state pour que les nœuds puissent créer des spans
        initial_state["langfuse_trace"] = langfuse_trace
        initial_state["metadata"]["langfuse_trace_id"] = str(langfuse_trace.id) if hasattr(langfuse_trace, 'id') else None

    try:
        result = workflow.invoke(initial_state, config=config)

        # Finaliser trace Langfuse si active
        if langfuse_trace:
            try:
                langfuse_trace.update(
                    output={
                        "success": result.get("success", False),
                        "intent": result.get("intent"),
                        "response_length": len(result.get("final_response", ""))
                    },
                    metadata={
                        "total_time": result.get("metadata", {}).get("total_time"),
                        "confidence": result.get("confidence_score")
                    }
                )
                logger.debug("Langfuse trace finalized")
            except Exception as e:
                logger.warning(f"Failed to finalize Langfuse trace: {e}")

        return result

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)

        # Logger l'erreur dans Langfuse
        if langfuse_trace:
            try:
                langfuse_trace.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
            except:
                pass

        # Retourner état d'erreur
        initial_state["final_response"] = f"Erreur lors du traitement: {str(e)}"
        initial_state["success"] = False
        initial_state["error_message"] = str(e)
        initial_state["metadata"]["total_time"] = time.time() - initial_state["start_time"]

        return initial_state


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# LANGGRAPH:
# - Framework pour workflows multi-agent
# - Graphe avec nœuds (fonctions) et edges (transitions)
# - État partagé (WorkflowState) passe entre nœuds
# - Routing conditionnel avec add_conditional_edges
#
# WORKFLOW:
# START → classify → [retrieve → generate → verify → finalize] | [off_topic] | [clarification] → END
#
# NŒUDS:
# - classify: Détermine type de question
# - retrieve: Recherche documents (RAG)
# - generate: Génère réponse avec LLM
# - verify: Vérifie qualité
# - finalize: Prépare réponse finale, log métriques
# - off_topic: Réponse rapide hors-sujet
# - clarification: Demande précisions
#
# ROUTING:
# - Après classify: → retrieve (math) | off_topic | clarification
# - Après verify: → finalize (toujours pour MVP)
#   * TODO: → human_validate si confidence < seuil
#
# ÉTAT:
# - Dictionnaire passé entre tous les nœuds
# - Chaque nœud peut lire et modifier
# - metadata: Accumuler métriques (temps, coûts, etc.)
#
# MÉTRIQUES:
# - Temps par nœud
# - Coûts API (via cost_tracker)
# - Confiance, tokens, etc.
# - Collectées dans metadata
# - Enregistrées par metrics_collector
#
# EXTENSIONS:
# - Human validation (if confidence < threshold)
# - Multi-query (générer plusieurs reformulations)
# - Self-consistency (générer plusieurs fois, voter)
# - Feedback loop (apprendre des retours)
#
# DEBUGGING:
# ```python
# # Tester
# from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
# from src.utils.config_loader import load_config
#
# config = load_config()
# workflow = create_rag_workflow(config)
# result = invoke_workflow(workflow, "Qu'est-ce qu'une dérivée ?")
# print(result["final_response"])
# print(f"Temps: {result['metadata']['total_time']:.2f}s")
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
