"""
Calcul et agrégation de métriques système.

Ce module gère:
- Temps d'exécution par agent
- Scores de confiance
- Taux de réussite/échec
- Sources les plus utilisées
- Export de rapports de performance

Usage:
    from src.utils.metrics import MetricsCollector

    metrics = MetricsCollector()
    with metrics.track_time("retrieval"):
        results = search_vectors(query)
    metrics.add_metric("confidence_score", 0.95)
    print(metrics.get_summary())
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import json
import statistics

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses pour métriques
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimingMetric:
    """Métrique de temps d'exécution."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def complete(self) -> float:
        """
        Marque la métrique comme complétée et calcule la durée.

        Returns:
            Durée en secondes
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


@dataclass
class QueryMetrics:
    """Métriques pour une requête utilisateur."""
    query_id: str
    timestamp: str
    question: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    verification_time: Optional[float] = None
    total_time: Optional[float] = None
    documents_retrieved: int = 0
    response_length: int = 0
    confidence_score: Optional[float] = None
    sources_cited: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# MetricsCollector
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Collecteur de métriques pour le système RAG.

    Accumule et agrège des métriques de performance, qualité, et usage.

    Example:
        >>> metrics = MetricsCollector()
        >>> with metrics.track_time("retrieval"):
        ...     docs = search(query)
        >>> metrics.add_metric("confidence", 0.92)
        >>> print(metrics.get_summary())
    """

    def __init__(self):
        """Initialise le collecteur."""
        # Timings par opération
        self.timings: Dict[str, List[float]] = defaultdict(list)

        # Métriques scalaires (confidence, etc.)
        self.scalars: Dict[str, List[float]] = defaultdict(list)

        # Compteurs
        self.counters: Dict[str, int] = defaultdict(int)

        # Queries complètes
        self.queries: List[QueryMetrics] = []

        # Sources citées (pour stats)
        self.sources_counter: Counter = Counter()

        # Session info
        self.session_start = datetime.now()

        logger.info("MetricsCollector initialized")

    def track_time(self, operation: str) -> "TimeTracker":
        """
        Context manager pour mesurer le temps d'une opération.

        Args:
            operation: Nom de l'opération (retrieval, generation, etc.)

        Returns:
            Context manager TimeTracker

        Example:
            >>> metrics = MetricsCollector()
            >>> with metrics.track_time("search"):
            ...     results = search_vectors(query)
            >>> print(f"Search took {metrics.get_avg_time('search'):.3f}s")
        """
        return TimeTracker(self, operation)

    def add_timing(self, operation: str, duration: float) -> None:
        """
        Ajoute une mesure de temps manuellement.

        Args:
            operation: Nom de l'opération
            duration: Durée en secondes
        """
        self.timings[operation].append(duration)
        logger.debug(
            f"Timing recorded: {operation} = {duration:.3f}s",
            extra={"operation": operation, "duration": duration}
        )

    def add_metric(self, name: str, value: float) -> None:
        """
        Ajoute une métrique scalaire.

        Args:
            name: Nom de la métrique (confidence_score, similarity, etc.)
            value: Valeur (généralement entre 0 et 1)
        """
        self.scalars[name].append(value)
        logger.debug(
            f"Metric recorded: {name} = {value:.3f}",
            extra={"metric": name, "value": value}
        )

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """
        Incrémente un compteur.

        Args:
            name: Nom du compteur (queries, errors, etc.)
            amount: Montant à incrémenter
        """
        self.counters[name] += amount

    def add_query(self, query_metrics: QueryMetrics) -> None:
        """
        Enregistre les métriques d'une requête complète.

        Args:
            query_metrics: Objet QueryMetrics avec toutes les infos
        """
        self.queries.append(query_metrics)

        # Mettre à jour compteurs
        self.increment_counter("total_queries")
        if query_metrics.success:
            self.increment_counter("successful_queries")
        else:
            self.increment_counter("failed_queries")

        # Mettre à jour sources
        for source in query_metrics.sources_cited:
            self.sources_counter[source] += 1

        logger.info(
            f"Query recorded: {query_metrics.query_id}",
            extra={
                "query_id": query_metrics.query_id,
                "success": query_metrics.success,
                "total_time": query_metrics.total_time
            }
        )

    def get_avg_time(self, operation: str) -> Optional[float]:
        """
        Retourne le temps moyen d'une opération.

        Args:
            operation: Nom de l'opération

        Returns:
            Temps moyen en secondes, ou None si aucune mesure
        """
        if operation not in self.timings or not self.timings[operation]:
            return None
        return statistics.mean(self.timings[operation])

    def get_median_time(self, operation: str) -> Optional[float]:
        """
        Retourne le temps médian d'une opération.

        Args:
            operation: Nom de l'opération

        Returns:
            Temps médian en secondes, ou None si aucune mesure
        """
        if operation not in self.timings or not self.timings[operation]:
            return None
        return statistics.median(self.timings[operation])

    def get_avg_metric(self, name: str) -> Optional[float]:
        """
        Retourne la moyenne d'une métrique scalaire.

        Args:
            name: Nom de la métrique

        Returns:
            Moyenne, ou None si aucune mesure
        """
        if name not in self.scalars or not self.scalars[name]:
            return None
        return statistics.mean(self.scalars[name])

    def get_success_rate(self) -> float:
        """
        Calcule le taux de réussite des requêtes.

        Returns:
            Taux de réussite (0.0 à 1.0)
        """
        total = self.counters.get("total_queries", 0)
        if total == 0:
            return 0.0

        successful = self.counters.get("successful_queries", 0)
        return successful / total

    def get_top_sources(self, n: int = 10) -> List[tuple]:
        """
        Retourne les sources les plus citées.

        Args:
            n: Nombre de sources à retourner

        Returns:
            Liste de tuples (source, count) triée par count décroissant

        Example:
            >>> top_sources = metrics.get_top_sources(5)
            >>> for source, count in top_sources:
            ...     print(f"{source}: {count} fois")
            Analyse_L2.pdf: 15 fois
            Calcul_L3.pdf: 12 fois
        """
        return self.sources_counter.most_common(n)

    def get_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé complet des métriques.

        Returns:
            Dictionnaire avec toutes les statistiques agrégées

        Example:
            >>> summary = metrics.get_summary()
            >>> print(f"Avg query time: {summary['avg_query_time']:.2f}s")
            >>> print(f"Success rate: {summary['success_rate']:.1%}")
        """
        # Temps moyens par opération
        avg_times = {
            op: self.get_avg_time(op)
            for op in self.timings.keys()
        }

        # Métriques moyennes
        avg_metrics = {
            name: self.get_avg_metric(name)
            for name in self.scalars.keys()
        }

        # Temps total moyen de requête
        avg_query_time = None
        if self.queries:
            query_times = [q.total_time for q in self.queries if q.total_time]
            if query_times:
                avg_query_time = statistics.mean(query_times)

        # Confidence moyenne
        avg_confidence = None
        if self.queries:
            confidences = [q.confidence_score for q in self.queries if q.confidence_score]
            if confidences:
                avg_confidence = statistics.mean(confidences)

        return {
            # Compteurs
            "total_queries": self.counters.get("total_queries", 0),
            "successful_queries": self.counters.get("successful_queries", 0),
            "failed_queries": self.counters.get("failed_queries", 0),
            "success_rate": self.get_success_rate(),

            # Temps
            "avg_query_time": avg_query_time,
            "avg_times_by_operation": avg_times,

            # Métriques
            "avg_confidence": avg_confidence,
            "avg_metrics": avg_metrics,

            # Sources
            "unique_sources_used": len(self.sources_counter),
            "top_sources": self.get_top_sources(10),

            # Session
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
        }

    def export_report(self, output_path: Optional[Path] = None) -> str:
        """
        Exporte un rapport détaillé en JSON.

        Args:
            output_path: Chemin du fichier de sortie (optionnel)

        Returns:
            JSON string du rapport

        Example:
            >>> metrics.export_report(Path("data/logs/metrics_report.json"))
        """
        report = {
            "summary": self.get_summary(),
            "queries": [
                {
                    "query_id": q.query_id,
                    "timestamp": q.timestamp,
                    "question": q.question,
                    "intent": q.intent,
                    "total_time": q.total_time,
                    "confidence_score": q.confidence_score,
                    "success": q.success,
                    "sources_cited": q.sources_cited,
                }
                for q in self.queries
            ],
            "timings": {
                op: {
                    "count": len(times),
                    "avg": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                }
                for op, times in self.timings.items()
                if times
            },
            "generated_at": datetime.now().isoformat()
        }

        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_json)
            logger.info(f"Metrics report exported to {output_path}")

        return report_json

    def reset(self) -> None:
        """Reset toutes les métriques."""
        self.timings.clear()
        self.scalars.clear()
        self.counters.clear()
        self.queries.clear()
        self.sources_counter.clear()
        self.session_start = datetime.now()
        logger.info("Metrics reset")


class TimeTracker:
    """
    Context manager pour tracking de temps.

    Example:
        >>> metrics = MetricsCollector()
        >>> with TimeTracker(metrics, "search"):
        ...     results = search()
    """

    def __init__(self, collector: MetricsCollector, operation: str):
        """
        Args:
            collector: MetricsCollector à mettre à jour
            operation: Nom de l'opération
        """
        self.collector = collector
        self.operation = operation
        self.start_time = None
        self.duration = None

    def __enter__(self):
        """Démarre le timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Arrête le timer et enregistre."""
        self.duration = time.time() - self.start_time
        self.collector.add_timing(self.operation, self.duration)


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def create_query_metrics(
    question: str,
    workflow_state: Dict[str, Any]
) -> QueryMetrics:
    """
    Crée un objet QueryMetrics depuis l'état du workflow.

    Args:
        question: Question de l'utilisateur
        workflow_state: État final du workflow LangGraph

    Returns:
        QueryMetrics complet

    Example:
        >>> final_state = workflow.invoke({"question": "..."})
        >>> query_metrics = create_query_metrics("...", final_state)
        >>> metrics.add_query(query_metrics)
    """
    return QueryMetrics(
        query_id=workflow_state.get("query_id", datetime.now().strftime("%Y%m%d%H%M%S")),
        timestamp=datetime.now().isoformat(),
        question=question,
        intent=workflow_state.get("intent"),
        intent_confidence=workflow_state.get("intent_confidence"),
        retrieval_time=workflow_state.get("retrieval_time"),
        generation_time=workflow_state.get("generation_time"),
        verification_time=workflow_state.get("verification_time"),
        total_time=workflow_state.get("total_time"),
        documents_retrieved=len(workflow_state.get("retrieved_docs") or []),
        response_length=len(workflow_state.get("final_response") or ""),
        confidence_score=workflow_state.get("confidence_score"),
        sources_cited=workflow_state.get("sources_cited") or [],
        success=workflow_state.get("success", True),
        error_message=workflow_state.get("error_message"),
        metadata=workflow_state.get("metadata") or {}
    )


def format_metrics_table(metrics: MetricsCollector) -> str:
    """
    Formate les métriques en tableau lisible.

    Args:
        metrics: MetricsCollector

    Returns:
        String formatée avec tableau

    Example:
        >>> print(format_metrics_table(metrics))
        ╔══════════════════════════════════════╗
        ║ Performance Metrics                  ║
        ╠══════════════════════════════════════╣
        ║ Total queries: 42                    ║
        ║ Success rate: 95.2%                  ║
        ║ Avg query time: 2.3s                 ║
        ╚══════════════════════════════════════╝
    """
    summary = metrics.get_summary()

    lines = [
        "╔══════════════════════════════════════╗",
        "║ Performance Metrics                  ║",
        "╠══════════════════════════════════════╣",
        f"║ Total queries: {summary['total_queries']:<22} ║",
        f"║ Success rate: {summary['success_rate']:.1%:<23} ║",
    ]

    if summary['avg_query_time']:
        lines.append(f"║ Avg query time: {summary['avg_query_time']:.2f}s{' '*(17-len(f'{summary['avg_query_time']:.2f}'))} ║")

    if summary['avg_confidence']:
        lines.append(f"║ Avg confidence: {summary['avg_confidence']:.2f}{' '*(18-len(f'{summary['avg_confidence']:.2f}'))} ║")

    lines.append("╚══════════════════════════════════════╝")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# MÉTRIQUES COLLECTÉES:
# 1. Timings: Temps d'exécution par opération
#    - retrieval, generation, verification, total_query
# 2. Scalars: Métriques continues (0-1 généralement)
#    - confidence_score, similarity_score
# 3. Counters: Compteurs simples
#    - total_queries, successful_queries, failed_queries
# 4. Queries: Métriques complètes par requête
# 5. Sources: Fréquence d'utilisation des sources
#
# USAGE RECOMMANDÉ:
# 1. Créer MetricsCollector au démarrage (singleton ou dans session)
# 2. Utiliser track_time() pour opérations importantes
# 3. Ajouter QueryMetrics après chaque requête complète
# 4. Afficher summary() dans interface web
# 5. Exporter rapport en fin de session
#
# STATISTIQUES:
# - Moyenne (mean): Représentative si distribution normale
# - Médiane (median): Plus robuste aux outliers
# - Pour temps: médiane souvent plus pertinente
#
# AFFICHAGE INTERFACE WEB:
# - Utiliser get_summary() pour métriques sidebar
# - format_metrics_table() pour affichage texte
# - export_report() pour génération graphiques (avec pandas/plotly)
#
# AGRÉGATION:
# - Par session: reset() entre sessions
# - Globale: Persister métriques en DB pour historique
# - Par utilisateur: Ajouter user_id dans QueryMetrics
#
# OPTIMISATION:
# - Si trop de queries, limiter taille de self.queries (FIFO)
# - Pour production: envoyer métriques à système externe
#   (Prometheus, Datadog, CloudWatch, etc.)
#
# EXTENSION:
# - Ajouter métriques custom: add_metric(), increment_counter()
# - Intégrer avec Langfuse pour observabilité LLM
# - Créer dashboards (Grafana, Streamlit, etc.)
#
# ═══════════════════════════════════════════════════════════════════════════════
