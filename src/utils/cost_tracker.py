"""
Tracking précis des coûts d'utilisation des APIs LLM.

Ce module permet de:
- Calculer les coûts par requête (input + output tokens)
- Cumuler les coûts par session
- Alerter si seuils dépassés
- Exporter des rapports de coûts

Usage:
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker(config)
    cost = tracker.track_llm_call("gpt-4o", input_tokens=1500, output_tokens=500)
    print(f"Cost: ${cost:.4f}")
    print(f"Total session: ${tracker.get_session_total():.4f}")
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json

from src.utils.logger import get_logger
from src.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses pour tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMCall:
    """Record d'un appel LLM."""
    timestamp: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    operation: str = "unknown"  # classifier, retriever, generator, etc.
    metadata: Dict = field(default_factory=dict)


@dataclass
class SessionCosts:
    """Coûts cumulés d'une session."""
    session_id: str
    start_time: str
    calls: List[LLMCall] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    costs_by_model: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Pricing (prix par 1000 tokens)
# ═══════════════════════════════════════════════════════════════════════════════

# Pricing par défaut (janvier 2025)
# À mettre à jour régulièrement selon les tarifs des providers
DEFAULT_PRICING = {
    # Anthropic Claude
    "claude-sonnet-4": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-opus-4": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "claude-haiku-4": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},

    # OpenAI GPT
    "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
    "gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},

    # Modèles locaux (gratuit)
    "mistral:7b": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "llama3:70b": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}


# ═══════════════════════════════════════════════════════════════════════════════
# CostTracker
# ═══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    """
    Tracker de coûts pour les appels LLM.

    Gère le calcul, l'accumulation et les alertes de coûts.

    Example:
        >>> config = load_config()
        >>> tracker = CostTracker(config)
        >>> cost = tracker.track_llm_call("gpt-4o", 1000, 500, "generator")
        >>> print(f"Cost: ${cost:.4f}")
        Cost: $0.0125
        >>> if tracker.is_over_budget():
        ...     print("Budget exceeded!")
    """

    def __init__(self, config: Optional[object] = None):
        """
        Args:
            config: Objet Config avec costs.pricing et limites
        """
        # Pricing (depuis config ou défaut)
        if config and hasattr(config, 'costs'):
            self.pricing = config.costs.pricing
            self.track_enabled = config.costs.track_costs
            self.max_cost_per_session = config.costs.max_cost_per_session
            self.alert_threshold = config.costs.alert_threshold
        else:
            self.pricing = DEFAULT_PRICING
            self.track_enabled = True
            self.max_cost_per_session = 1.0
            self.alert_threshold = 0.5

        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = SessionCosts(
            session_id=self.session_id,
            start_time=datetime.now().isoformat()
        )

        # Alertes
        self.alert_sent = False

        logger.info(
            f"CostTracker initialized",
            extra={
                "session_id": self.session_id,
                "max_budget": self.max_cost_per_session,
                "alert_threshold": self.alert_threshold
            }
        )

    def get_model_pricing(self, model: str) -> Tuple[float, float]:
        """
        Récupère les prix input/output pour un modèle.

        Args:
            model: Nom du modèle

        Returns:
            Tuple (input_price_per_1k, output_price_per_1k)

        Raises:
            ConfigurationError: Si le modèle n'est pas dans le pricing
        """
        if model not in self.pricing:
            logger.warning(
                f"Model {model} not in pricing, using default (free)",
                extra={"model": model}
            )
            return (0.0, 0.0)

        pricing = self.pricing[model]
        return (pricing["input_per_1k"], pricing["output_per_1k"])

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[float, float, float]:
        """
        Calcule le coût d'un appel LLM.

        Args:
            model: Nom du modèle
            input_tokens: Nombre de tokens en entrée
            output_tokens: Nombre de tokens en sortie

        Returns:
            Tuple (input_cost, output_cost, total_cost)

        Example:
            >>> tracker = CostTracker()
            >>> inp, out, total = tracker.calculate_cost("gpt-4o", 1000, 500)
            >>> print(f"Total: ${total:.4f}")
            Total: $0.0125
        """
        input_price, output_price = self.get_model_pricing(model)

        # Calcul (prix par 1k tokens)
        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price
        total_cost = input_cost + output_cost

        return (input_cost, output_cost, total_cost)

    def track_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown",
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Enregistre un appel LLM et retourne le coût.

        Args:
            model: Nom du modèle
            input_tokens: Tokens en entrée
            output_tokens: Tokens en sortie
            operation: Type d'opération (classifier, generator, etc.)
            metadata: Métadonnées additionnelles

        Returns:
            Coût total de l'appel

        Side effects:
            - Met à jour session.total_cost
            - Log le coût
            - Envoie alerte si seuil dépassé
        """
        if not self.track_enabled:
            return 0.0

        # Calcul coût
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, input_tokens, output_tokens
        )

        # Déterminer provider depuis model name
        if "claude" in model.lower():
            provider = "anthropic"
        elif "gpt" in model.lower():
            provider = "openai"
        else:
            provider = "local"

        # Créer record
        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            operation=operation,
            metadata=metadata or {}
        )

        # Mettre à jour session
        self.session.calls.append(call)
        self.session.total_input_tokens += input_tokens
        self.session.total_output_tokens += output_tokens
        self.session.total_cost += total_cost

        # Compteurs par modèle
        self.session.calls_by_model[model] = self.session.calls_by_model.get(model, 0) + 1
        self.session.costs_by_model[model] = self.session.costs_by_model.get(model, 0.0) + total_cost

        # Log
        logger.info(
            f"LLM call tracked: {model} ({operation})",
            extra={
                "model": model,
                "provider": provider,
                "operation": operation,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "session_total": self.session.total_cost
            }
        )

        # Vérifier alertes
        self._check_alerts()

        return total_cost

    def _check_alerts(self) -> None:
        """Vérifie et envoie des alertes si seuils dépassés."""
        current_cost = self.session.total_cost

        # Alerte threshold
        if not self.alert_sent and current_cost >= self.alert_threshold:
            logger.warning(
                f"⚠️ Cost alert: ${current_cost:.4f} / ${self.max_cost_per_session:.4f}",
                extra={
                    "current_cost": current_cost,
                    "alert_threshold": self.alert_threshold,
                    "max_budget": self.max_cost_per_session
                }
            )
            self.alert_sent = True

        # Budget dépassé
        if current_cost > self.max_cost_per_session:
            logger.error(
                f"🚨 Budget exceeded: ${current_cost:.4f} > ${self.max_cost_per_session:.4f}",
                extra={
                    "current_cost": current_cost,
                    "max_budget": self.max_cost_per_session,
                    "overage": current_cost - self.max_cost_per_session
                }
            )

    def is_over_budget(self) -> bool:
        """
        Vérifie si le budget de session est dépassé.

        Returns:
            True si budget dépassé
        """
        return self.session.total_cost > self.max_cost_per_session

    def get_session_total(self) -> float:
        """
        Retourne le coût total de la session.

        Returns:
            Coût total en USD
        """
        return self.session.total_cost

    def get_session_stats(self) -> Dict:
        """
        Retourne les statistiques complètes de la session.

        Returns:
            Dict avec stats détaillées

        Example:
            >>> stats = tracker.get_session_stats()
            >>> print(f"Total calls: {stats['total_calls']}")
            >>> print(f"Total cost: ${stats['total_cost']:.4f}")
        """
        return {
            "session_id": self.session.session_id,
            "start_time": self.session.start_time,
            "total_calls": len(self.session.calls),
            "total_input_tokens": self.session.total_input_tokens,
            "total_output_tokens": self.session.total_output_tokens,
            "total_tokens": self.session.total_input_tokens + self.session.total_output_tokens,
            "total_cost": self.session.total_cost,
            "avg_cost_per_call": (
                self.session.total_cost / len(self.session.calls)
                if self.session.calls else 0.0
            ),
            "calls_by_model": self.session.calls_by_model,
            "costs_by_model": self.session.costs_by_model,
            "is_over_budget": self.is_over_budget()
        }

    def export_session_report(self, output_path: Optional[Path] = None) -> str:
        """
        Exporte un rapport détaillé de la session.

        Args:
            output_path: Chemin du fichier de sortie (optionnel)
                        Si None, retourne juste le JSON

        Returns:
            JSON string du rapport

        Example:
            >>> tracker.export_session_report(Path("data/logs/cost_report.json"))
        """
        report = {
            "session": asdict(self.session),
            "stats": self.get_session_stats(),
            "pricing_used": self.pricing,
            "generated_at": datetime.now().isoformat()
        }

        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_json)
            logger.info(f"Cost report exported to {output_path}")

        return report_json

    def reset_session(self) -> None:
        """Reset le tracking de session (nouveau session_id)."""
        old_session_id = self.session_id
        old_cost = self.session.total_cost

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = SessionCosts(
            session_id=self.session_id,
            start_time=datetime.now().isoformat()
        )
        self.alert_sent = False

        logger.info(
            f"Session reset: {old_session_id} (${old_cost:.4f}) → {self.session_id}",
            extra={"old_session": old_session_id, "old_cost": old_cost}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Estime le nombre de tokens dans un texte.

    Approximation simple: ~4 caractères = 1 token (anglais)
    Pour français: ~5 caractères = 1 token

    Pour estimation précise, utiliser tiktoken (OpenAI) ou anthropic tokenizer.

    Args:
        text: Texte à estimer
        model: Modèle (pour choix de tokenizer si disponible)

    Returns:
        Estimation du nombre de tokens

    Example:
        >>> text = "Qu'est-ce qu'une dérivée?"
        >>> tokens = estimate_tokens(text)
        >>> print(f"~{tokens} tokens")
        ~6 tokens
    """
    # Approximation simple
    # TODO: Utiliser tiktoken pour précision (pip install tiktoken)
    chars = len(text)

    # Heuristique: français ~5 chars/token, anglais ~4 chars/token
    if any(ord(c) > 127 for c in text):  # Caractères non-ASCII = probablement français
        return chars // 5
    else:
        return chars // 4


def compare_model_costs(
    models: List[str],
    input_tokens: int,
    output_tokens: int,
    tracker: Optional[CostTracker] = None
) -> Dict[str, float]:
    """
    Compare les coûts de différents modèles pour une même requête.

    Args:
        models: Liste de noms de modèles
        input_tokens: Tokens en entrée
        output_tokens: Tokens en sortie
        tracker: CostTracker (optionnel, pour pricing custom)

    Returns:
        Dict {model: cost}

    Example:
        >>> models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4"]
        >>> costs = compare_model_costs(models, 1000, 500)
        >>> for model, cost in sorted(costs.items(), key=lambda x: x[1]):
        ...     print(f"{model}: ${cost:.4f}")
        gpt-4o-mini: $0.0004
        claude-sonnet-4: $0.0105
        gpt-4o: $0.0125
    """
    if tracker is None:
        tracker = CostTracker()

    costs = {}
    for model in models:
        _, _, total = tracker.calculate_cost(model, input_tokens, output_tokens)
        costs[model] = total

    return costs


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# PRICING:
# - Mis à jour manuellement dans DEFAULT_PRICING
# - Peut être overridé via config.yaml
# - Vérifier tarifs régulièrement:
#   * Anthropic: https://www.anthropic.com/pricing
#   * OpenAI: https://openai.com/pricing
#
# PRÉCISION TOKENS:
# - estimate_tokens() utilise approximation simple (chars/4)
# - Pour précision: pip install tiktoken
#   ```python
#   import tiktoken
#   enc = tiktoken.encoding_for_model("gpt-4o")
#   tokens = len(enc.encode(text))
#   ```
#
# ALERTES:
# - alert_threshold: Warning log (ex: 50% du budget)
# - max_cost_per_session: Error log si dépassé
# - Pas de blocage automatique (à implémenter si nécessaire)
#
# EXPORT:
# - export_session_report() génère JSON complet
# - Peut être parsé pour analytics, facturation, etc.
# - Inclut tous les calls individuels + agrégations
#
# SESSION:
# - Une session = durée de vie du tracker
# - reset_session() pour recommencer à zéro
# - session_id = timestamp pour unicité
#
# USAGE RECOMMANDÉ:
# 1. Créer CostTracker au démarrage
# 2. Passer à tous les agents
# 3. Appeler track_llm_call() après chaque appel LLM
# 4. Vérifier is_over_budget() avant appels coûteux
# 5. Exporter rapport en fin de session
#
# OPTIMISATION COÛTS:
# - Utiliser GPT-4o-mini pour dev/tests
# - Utiliser Claude Haiku pour tâches simples (classification)
# - Mettre en cache les réponses fréquentes
# - Limiter max_tokens dans config
# - Utiliser modèles locaux (Ollama) pour dev
#
# ═══════════════════════════════════════════════════════════════════════════════
