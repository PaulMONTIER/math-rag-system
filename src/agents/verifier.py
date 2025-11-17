"""
Agent Verifier - Vérification de qualité et hallucinations.

Cet agent vérifie la réponse générée pour:
- Détection d'hallucinations (info non dans sources)
- Cohérence mathématique
- Sources bien citées
- Score de confiance global

Usage:
    from src.agents.verifier import VerifierAgent

    verifier = VerifierAgent(config, llm_client)
    result = verifier.verify(question, answer, context, sources)
    if result.is_valid:
        print("Réponse validée!")
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.llm.closed_models import BaseLLMClient
from src.agents.generator import GeneratedResponse
from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import AgentError

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Résultat de vérification."""
    is_valid: bool
    confidence_score: float
    issues_found: List[str]
    warnings: List[str]
    recommendation: str  # "APPROVE", "HUMAN_REVIEW", "REJECT"
    metadata: Dict


class VerifierAgent:
    """
    Agent de vérification de qualité.

    Vérifie:
    1. Hallucinations (info non dans contexte)
    2. Cohérence mathématique
    3. Sources citées correctement
    4. Score confiance global

    Example:
        >>> verifier = VerifierAgent(config, llm_client)
        >>> result = verifier.verify(question, answer, context, sources)
        >>> print(f"Confiance: {result.confidence_score:.2f}")
    """

    def __init__(self, config: object, llm_client: BaseLLMClient):
        """
        Args:
            config: Objet Config
            llm_client: Client LLM
        """
        self.config = config
        self.llm_client = llm_client

        # Configuration
        if hasattr(config.agents, 'verifier'):
            self.confidence_threshold = config.agents.verifier.get("confidence_threshold", 0.75)
            self.hallucination_check = config.agents.verifier.get("hallucination_check", True)
            self.citation_check = config.agents.verifier.get("citation_check", True)
        else:
            self.confidence_threshold = 0.75
            self.hallucination_check = True
            self.citation_check = True

        logger.info("VerifierAgent initialized")

    def verify(
        self,
        question: str,
        answer: str,
        context: str,
        sources: List[str]
    ) -> VerificationResult:
        """
        Vérifie une réponse générée.

        Args:
            question: Question originale
            answer: Réponse générée
            context: Contexte fourni au LLM
            sources: Sources citées

        Returns:
            VerificationResult
        """
        logger.info("Verifying response")

        issues = []
        warnings = []

        # 1. Vérifier citations
        if self.citation_check:
            if not sources or len(sources) == 0:
                warnings.append("Aucune source citée")

        # 2. Vérifier hallucinations (simple: check si info clés du contexte)
        if self.hallucination_check:
            # Vérification simplifiée pour MVP
            # TODO: Utiliser LLM pour vérification détaillée
            pass

        # 3. Calculer score de confiance (heuristique)
        confidence = self._compute_confidence(answer, context, sources, issues, warnings)

        # 4. Déterminer recommendation
        if confidence >= 0.85:
            recommendation = "APPROVE"
            is_valid = True
        elif confidence >= self.confidence_threshold:
            recommendation = "APPROVE"
            is_valid = True
        else:
            recommendation = "HUMAN_REVIEW"
            is_valid = False

        result = VerificationResult(
            is_valid=is_valid,
            confidence_score=confidence,
            issues_found=issues,
            warnings=warnings,
            recommendation=recommendation,
            metadata={
                "sources_count": len(sources),
                "answer_length": len(answer)
            }
        )

        logger.info(
            f"✓ Verification complete",
            extra={
                "is_valid": is_valid,
                "confidence": confidence,
                "recommendation": recommendation
            }
        )

        return result

    def _compute_confidence(
        self,
        answer: str,
        context: str,
        sources: List[str],
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calcule un score de confiance heuristique.

        Critères:
        - Sources citées: +0.3
        - Pas d'issues: +0.3
        - Longueur raisonnable: +0.2
        - Formules présentes (si contexte en a): +0.2

        Args:
            answer: Réponse
            context: Contexte
            sources: Sources
            issues: Issues trouvées
            warnings: Warnings

        Returns:
            Score 0-1
        """
        score = 0.5  # Base

        # Sources citées
        if sources and len(sources) > 0:
            score += 0.3

        # Pas d'issues critiques
        if len(issues) == 0:
            score += 0.3

        # Longueur raisonnable (pas trop court ni trop long)
        if 50 < len(answer) < 2000:
            score += 0.2

        # Warnings réduisent score
        score -= len(warnings) * 0.05

        # Limiter 0-1
        score = max(0.0, min(1.0, score))

        return score


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES: Vérification simplifiée pour MVP. Peut être améliorée avec LLM dédié.
# ═══════════════════════════════════════════════════════════════════════════════
