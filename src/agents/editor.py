"""
Agent Editor - Review et amélioration des réponses générées.

Cet agent analyse la réponse générée et peut :
- Suggérer des améliorations
- Identifier des incohérences
- Vérifier la complétude
- Proposer des clarifications

Usage:
    from src.agents.editor import EditorAgent, EditorDecision

    agent = EditorAgent(config, llm_client)
    decision = agent.review(question, generated_answer, context)
    if decision.needs_revision:
        improved = decision.improved_answer
"""

from typing import Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EditorDecision:
    """Décision de l'éditeur après review."""
    needs_revision: bool
    improved_answer: Optional[str]
    issues_found: list[str]
    suggestions: list[str]
    quality_score: float
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════════════
# Editor Agent
# ═══════════════════════════════════════════════════════════════════════════════

class EditorAgent:
    """
    Agent d'édition pour review et amélioration des réponses.

    Caractéristiques:
    - Vérifie la cohérence avec le contexte
    - Identifie les manques ou incohérences
    - Peut suggérer des améliorations
    - Évalue la qualité globale
    """

    def __init__(self, config: object, llm_client: object):
        """
        Initialise l'Editor Agent.

        Args:
            config: Configuration du système
            llm_client: Client LLM pour analyse
        """
        self.config = config
        self.llm_client = llm_client

        logger.info("EditorAgent initialized")

    def review(
        self,
        question: str,
        generated_answer: str,
        context: Optional[str] = None,
        sources: Optional[list] = None
    ) -> EditorDecision:
        """
        Review la réponse générée et suggère des améliorations.

        Args:
            question: Question originale
            generated_answer: Réponse générée par GeneratorAgent
            context: Contexte utilisé (optionnel)
            sources: Sources utilisées (optionnel)

        Returns:
            EditorDecision avec évaluation et suggestions

        Example:
            >>> agent = EditorAgent(config, llm_client)
            >>> decision = agent.review(
            ...     question="Qu'est-ce qu'une dérivée?",
            ...     generated_answer="La dérivée mesure...",
            ...     context="..."
            ... )
            >>> print(decision.quality_score)  # 0.85
        """
        logger.info(f"Reviewing answer for: {question[:50]}...")

        # Heuristiques simples pour MVP
        # TODO: Remplacer par appel LLM pour analyse fine

        issues_found = []
        suggestions = []
        quality_score = 1.0

        # Vérification 1: Longueur de la réponse
        if len(generated_answer) < 50:
            issues_found.append("Réponse très courte, pourrait manquer de détails")
            quality_score -= 0.2
            suggestions.append("Ajouter plus de détails et d'exemples")

        # Vérification 2: Présence de formules LaTeX (pour questions math)
        if "$" in question or "formule" in question.lower():
            if "$" not in generated_answer:
                issues_found.append("Question mathématique sans formule LaTeX")
                quality_score -= 0.1
                suggestions.append("Ajouter des formules mathématiques en LaTeX")

        # Vérification 3: Sources citées
        if sources and len(sources) == 0:
            issues_found.append("Aucune source citée")
            quality_score -= 0.15
            suggestions.append("Ajouter des références aux sources utilisées")

        # Vérification 4: Cohérence avec la question
        question_words = set(question.lower().split())
        answer_words = set(generated_answer.lower().split())
        overlap = len(question_words & answer_words) / max(len(question_words), 1)

        if overlap < 0.2:
            issues_found.append("Faible cohérence lexicale question-réponse")
            quality_score -= 0.15
            suggestions.append("S'assurer que la réponse adresse directement la question")

        # Vérification 5: Structure
        has_sections = any(marker in generated_answer for marker in ["##", "**", "1.", "2."])
        if len(generated_answer) > 500 and not has_sections:
            issues_found.append("Longue réponse sans structure claire")
            quality_score -= 0.1
            suggestions.append("Structurer la réponse avec des sections ou bullet points")

        # Décision finale
        needs_revision = quality_score < 0.75 and len(issues_found) > 0

        if needs_revision:
            improved_answer = self._improve_answer(
                generated_answer,
                issues_found,
                suggestions
            )
            reasoning = f"Révision nécessaire : {len(issues_found)} problèmes détectés"
        else:
            improved_answer = None
            reasoning = f"Qualité acceptable (score: {quality_score:.2f})"

        logger.info(f"  Quality score: {quality_score:.2f}")
        logger.info(f"  Needs revision: {needs_revision}")
        if issues_found:
            logger.info(f"  Issues: {', '.join(issues_found)}")

        return EditorDecision(
            needs_revision=needs_revision,
            improved_answer=improved_answer,
            issues_found=issues_found,
            suggestions=suggestions,
            quality_score=quality_score,
            reasoning=reasoning
        )

    def _improve_answer(
        self,
        answer: str,
        issues: list[str],
        suggestions: list[str]
    ) -> str:
        """
        Améliore la réponse basée sur les problèmes identifiés.

        Args:
            answer: Réponse originale
            issues: Liste des problèmes trouvés
            suggestions: Liste des suggestions

        Returns:
            Réponse améliorée
        """
        # Pour MVP : retourner la réponse originale avec note
        # TODO: Utiliser LLM pour régénération améliorée

        improvement_note = "\n\n---\n**Note d'amélioration suggérée** :\n"
        for i, suggestion in enumerate(suggestions, 1):
            improvement_note += f"{i}. {suggestion}\n"

        return answer + improvement_note


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# STRATÉGIE ACTUELLE (MVP):
# - Heuristiques de qualité simples
# - Pas de régénération automatique (pour éviter coûts API)
# - Suggestions plutôt que corrections
#
# AMÉLIORATIONS FUTURES:
# 1. Utiliser LLM pour analyse sémantique fine
# 2. Régénération automatique pour réponses insuffisantes
# 3. Vérification factuelle avec sources
# 4. Détection de hallucinations
# 5. Style et ton adapté au niveau étudiant
# 6. Cohérence avec le contexte RAG/Web
#
# CRITÈRES DE QUALITÉ:
# - Complétude (répond à toute la question)
# - Exactitude (cohérent avec sources)
# - Clarté (bien structuré, formules LaTeX)
# - Pédagogie (adapté au niveau)
# - Citations (sources référencées)
#
# INTÉGRATION WORKFLOW:
# - Utilisé après generation
# - Avant verification finale
# - Peut déclencher regénération si qualité < seuil
#
# ═══════════════════════════════════════════════════════════════════════════════
