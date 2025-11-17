"""
Agent Classifier - Classification de l'intention utilisateur.

Cet agent détermine le type de question pour router correctement dans le workflow:
- MATH_QUESTION: Question mathématique légitime → RAG
- OFF_TOPIC: Hors sujet → Réponse directe
- NEED_CLARIFICATION: Question ambiguë → Demander précisions
- GREETING: Salutation → Réponse amicale

Usage:
    from src.agents.classifier import ClassifierAgent

    classifier = ClassifierAgent(config, llm_client)
    result = classifier.classify("Qu'est-ce qu'une dérivée ?")
    print(result.intent)  # "MATH_QUESTION"
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

from src.llm.closed_models import BaseLLMClient
from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import AgentError

logger = get_logger(__name__)


class Intent(str, Enum):
    """Intentions possibles."""
    MATH_QUESTION = "MATH_QUESTION"
    CURRENT_EVENT = "CURRENT_EVENT"      # Actualités/événements mathématiques récents
    OFF_TOPIC = "OFF_TOPIC"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    GREETING = "GREETING"


@dataclass
class ClassificationResult:
    """Résultat de classification."""
    intent: Intent
    confidence: float
    reasoning: str
    suggested_action: str


class ClassifierAgent:
    """
    Agent de classification d'intention.

    Utilise un LLM avec prompt spécialisé pour classifier rapidement.

    Example:
        >>> classifier = ClassifierAgent(config, llm_client)
        >>> result = classifier.classify("Bonjour!")
        >>> print(result.intent)
        GREETING
    """

    def __init__(self, config: object, llm_client: BaseLLMClient):
        """
        Args:
            config: Objet Config
            llm_client: Client LLM
        """
        self.config = config
        self.llm_client = llm_client

        # Seuil de confiance
        if hasattr(config.agents, 'classifier'):
            self.confidence_threshold = config.agents.classifier.get("confidence_threshold", 0.8)
        else:
            self.confidence_threshold = 0.8

        logger.info("ClassifierAgent initialized")

    def classify(self, question: str) -> ClassificationResult:
        """
        Classifie une question.

        Args:
            question: Question utilisateur

        Returns:
            ClassificationResult
        """
        system_prompt = """Tu es un classificateur d'intentions pour un assistant mathématiques.

Classe la question dans l'une de ces catégories:
- MATH_QUESTION: Questions théoriques/conceptuelles sur les mathématiques:
  * Concepts classiques (dérivées, intégrales, topologie, etc.)
  * Résolution de problèmes
  * Démonstrations et preuves
  * Histoire établie des mathématiques
  * Questions sur des mathématiciens célèbres
- CURRENT_EVENT: Actualités et événements mathématiques RÉCENTS/ACTUELS:
  * "Actualités mathématiques 2024", "dernières avancées", "récentes découvertes"
  * Événements avec dates/années récentes (2024, 2025, "cette année")
  * Prix Nobel, médailles Fields récentes
  * "Nouvelles recherches", "découvertes récentes", "derniers résultats"
  * Mots-clés: actualité, récent, dernier, nouvelle, 2024, 2025, aujourd'hui
- OFF_TOPIC: Hors sujet (météo, cuisine, sport, politique générale, etc.)
- NEED_CLARIFICATION: Question trop vague ou ambiguë
- GREETING: Salutation simple (bonjour, merci, au revoir, etc.)

RÈGLE CRITIQUE: Si la question contient des indicateurs temporels récents (actualités,
2024, récent, dernier, nouveau, etc.), classe comme CURRENT_EVENT, pas MATH_QUESTION.

Réponds UNIQUEMENT au format JSON:
{
  "intent": "CURRENT_EVENT",
  "confidence": 0.95,
  "reasoning": "Question demande actualités 2024 (indicateur temporel)"
}"""

        user_prompt = f"Question: {question}"

        try:
            with log_performance(logger, "classification"):
                response = self.llm_client.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.1,  # Bas pour classification
                    max_tokens=150
                )

            # Parser JSON
            import json
            import re

            # Extraire JSON de la réponse
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                # Fallback: assumer MATH_QUESTION
                data = {
                    "intent": "MATH_QUESTION",
                    "confidence": 0.5,
                    "reasoning": "Unable to parse classification, defaulting to MATH_QUESTION"
                }

            intent = Intent(data.get("intent", "MATH_QUESTION"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            # Déterminer action
            if intent == Intent.MATH_QUESTION:
                action = "proceed_to_retrieval"
            elif intent == Intent.CURRENT_EVENT:
                action = "proceed_to_web_search"
            elif intent == Intent.OFF_TOPIC:
                action = "respond_off_topic"
            elif intent == Intent.NEED_CLARIFICATION:
                action = "ask_clarification"
            else:  # GREETING
                action = "respond_greeting"

            result = ClassificationResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                suggested_action=action
            )

            logger.info(
                f"✓ Classified as {intent.value}",
                extra={
                    "intent": intent.value,
                    "confidence": confidence,
                    "action": action
                }
            )

            return result

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)

            # Fallback sûr
            return ClassificationResult(
                intent=Intent.MATH_QUESTION,
                confidence=0.5,
                reasoning=f"Classification error: {e}",
                suggested_action="proceed_to_retrieval"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES: Utilisé GPT-4o-mini recommandé pour classification (rapide et pas cher)
# ═══════════════════════════════════════════════════════════════════════════════
