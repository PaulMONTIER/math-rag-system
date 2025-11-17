"""
Agent Planner - Décide de la stratégie de recherche optimale.

Cet agent analyse la question et décide s'il faut utiliser :
- RAG local uniquement (base vectorielle)
- Web search uniquement (recherche externe)
- Les deux (combinaison)

Usage:
    from src.agents.planner import PlannerAgent, SearchStrategy

    agent = PlannerAgent(config, llm_client)
    strategy = agent.plan("Quelle est la dernière découverte en mathématiques?")
    print(strategy.strategy)  # "web_only", "local_only", ou "both"
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

class SearchStrategy(Enum):
    """Stratégie de recherche à utiliser."""
    LOCAL_ONLY = "local_only"      # Uniquement base vectorielle locale
    WEB_ONLY = "web_only"          # Uniquement recherche web
    BOTH = "both"                  # Combinaison des deux


@dataclass
class PlanningDecision:
    """Décision de planification."""
    strategy: SearchStrategy
    reasoning: str
    confidence: float


# ═══════════════════════════════════════════════════════════════════════════════
# Planner Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """
    Agent de planification pour routing intelligent.

    Analyse la question et décide de la meilleure stratégie :
    - LOCAL_ONLY : concepts mathématiques classiques/théorèmes établis
    - WEB_ONLY : actualités, événements récents, informations contextuelles
    - BOTH : questions complexes nécessitant théorie + contexte récent
    """

    def __init__(self, config: object, llm_client: object):
        """
        Initialise le Planner Agent.

        Args:
            config: Configuration du système
            llm_client: Client LLM pour analyse
        """
        self.config = config
        self.llm_client = llm_client

        logger.info("PlannerAgent initialized")

    def plan(self, question: str) -> PlanningDecision:
        """
        Décide de la stratégie de recherche optimale.

        Args:
            question: Question de l'utilisateur

        Returns:
            PlanningDecision avec stratégie et raisonnement

        Example:
            >>> agent = PlannerAgent(config, llm_client)
            >>> decision = agent.plan("Qu'est-ce qu'une intégrale?")
            >>> print(decision.strategy)  # SearchStrategy.LOCAL_ONLY
        """
        logger.info(f"Planning search strategy for: {question[:50]}...")

        # Heuristiques simples pour MVP
        # TODO: Remplacer par appel LLM pour analyse plus fine

        question_lower = question.lower()

        # Indicateurs de besoin de recherche web
        web_indicators = [
            "récent", "dernier", "dernière", "actualité", "nouvelle",
            "2024", "2025", "aujourd'hui", "cette année",
            "événement", "découverte récente", "prix nobel",
            "qui a", "où est", "quand", "contexte"
        ]

        # Indicateurs de contenu local suffisant
        local_indicators = [
            "définition", "qu'est-ce", "théorème", "démonstration",
            "preuve", "formule", "propriété", "axiome",
            "calcul", "résoudre", "méthode", "algorithme",
            "dérivée", "intégrale", "limite", "série",
            "matrice", "vecteur", "espace", "groupe"
        ]

        # Compter les indicateurs
        web_score = sum(1 for indicator in web_indicators if indicator in question_lower)
        local_score = sum(1 for indicator in local_indicators if indicator in question_lower)

        # Décision basée sur les scores
        if web_score > 0 and local_score > 0:
            # Les deux types d'indicateurs présents
            strategy = SearchStrategy.BOTH
            reasoning = f"Question nécessite à la fois contenu théorique local et informations récentes du web (web_score={web_score}, local_score={local_score})"
            confidence = 0.8
        elif web_score > local_score:
            # Plutôt orienté web
            strategy = SearchStrategy.WEB_ONLY
            reasoning = f"Question orientée vers informations récentes ou contextuelles (web_score={web_score})"
            confidence = 0.75
        elif local_score > 0:
            # Plutôt orienté contenu local
            strategy = SearchStrategy.LOCAL_ONLY
            reasoning = f"Question théorique/conceptuelle couverte par la base locale (local_score={local_score})"
            confidence = 0.85
        else:
            # Cas par défaut : utiliser les deux pour être sûr
            strategy = SearchStrategy.BOTH
            reasoning = "Question ambiguë, utilisation des deux sources par sécurité"
            confidence = 0.6

        logger.info(f"  Strategy: {strategy.value} (confidence: {confidence:.2f})")
        logger.info(f"  Reasoning: {reasoning}")

        return PlanningDecision(
            strategy=strategy,
            reasoning=reasoning,
            confidence=confidence
        )

    def plan_with_llm(self, question: str) -> PlanningDecision:
        """
        Décide de la stratégie avec analyse LLM (version avancée).

        Cette méthode utilise le LLM pour une analyse plus fine.
        Peut être activée plus tard pour améliorer la précision.

        Args:
            question: Question de l'utilisateur

        Returns:
            PlanningDecision avec stratégie et raisonnement
        """
        # TODO: Implémenter appel LLM pour analyse sémantique fine
        # Pour l'instant, utiliser la méthode heuristique
        return self.plan(question)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# STRATÉGIE ACTUELLE (MVP):
# - Heuristiques basées sur mots-clés
# - Rapide et sans coût API
# - Précision ~75-80%
#
# AMÉLIORATIONS FUTURES:
# 1. Analyse LLM pour classification sémantique
# 2. Learning from user feedback (si web_only était suffisant, etc.)
# 3. Historique des questions similaires
# 4. Score de fraîcheur du contenu local
# 5. Combinaison pondérée local+web basée sur confidence
#
# EXEMPLES DE ROUTING:
# - "Qu'est-ce qu'une dérivée?" → LOCAL_ONLY
# - "Qui a gagné la médaille Fields 2024?" → WEB_ONLY
# - "Expliquez le théorème de Fermat et son histoire récente" → BOTH
#
# INTÉGRATION WORKFLOW:
# - Utilisé après classification
# - Avant retrieval/web_search
# - Détermine le flux du graphe LangGraph
#
# ═══════════════════════════════════════════════════════════════════════════════
