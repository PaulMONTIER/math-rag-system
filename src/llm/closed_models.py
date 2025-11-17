"""
Interfaces pour modèles LLM fermés (APIs Claude et GPT).

Ce module gère les appels aux APIs de:
- Anthropic Claude (recommandé pour mathématiques)
- OpenAI GPT (alternative performante)
- Fallback vers modèles locaux (Ollama)

Usage:
    from src.llm.closed_models import get_llm_client

    client = get_llm_client(config)
    response = client.generate(
        prompt="Qu'est-ce qu'une dérivée ?",
        system="Tu es un assistant mathématiques"
    )
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# LLM APIs
from openai import RateLimitError, APIError
from anthropic import Anthropic, RateLimitError as AnthropicRateLimit

# Standard OpenAI client (manual Langfuse tracing)
from openai import OpenAI
LANGFUSE_OPENAI_AVAILABLE = False  # Using manual tracing instead

from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import LLMAPIError, MissingAPIKeyError
from src.utils.cost_tracker import CostTracker
from src.utils.langfuse_context import get_current_langfuse_span

# Langfuse pour tracing LLM
try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Créer un décorateur no-op si Langfuse n'est pas disponible
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """
    Réponse d'un LLM.

    Attributes:
        content: Texte généré
        model: Modèle utilisé
        input_tokens: Tokens en entrée
        output_tokens: Tokens en sortie
        total_tokens: Total tokens
        cost: Coût en USD
        latency: Temps de génération (secondes)
        metadata: Métadonnées additionnelles
    """
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency: float
    metadata: Dict = None


# ═══════════════════════════════════════════════════════════════════════════════
# Base LLM Client
# ═══════════════════════════════════════════════════════════════════════════════

class BaseLLMClient:
    """
    Client LLM de base (interface commune).

    Toutes les implémentations (OpenAI, Anthropic, Ollama) héritent de cette classe.
    """

    def __init__(self, config: object, cost_tracker: Optional[CostTracker] = None):
        """
        Args:
            config: Objet Config
            cost_tracker: CostTracker pour tracking des coûts
        """
        self.config = config
        self.cost_tracker = cost_tracker or CostTracker(config)

        # Configuration commune
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.top_p = config.llm.top_p
        self.timeout = config.llm.timeout
        self.retry_attempts = config.llm.retry_attempts

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Génère une réponse (à implémenter par sous-classes).

        Args:
            prompt: Prompt utilisateur
            system: Prompt système (optionnel)
            **kwargs: Arguments additionnels

        Returns:
            LLMResponse
        """
        raise NotImplementedError("Subclass must implement generate()")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Exécute une fonction avec retry exponential backoff.

        Args:
            func: Fonction à exécuter
            *args, **kwargs: Arguments de la fonction

        Returns:
            Résultat de la fonction

        Raises:
            LLMAPIError: Si tous les retry échouent
        """
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)

            except (RateLimitError, AnthropicRateLimit) as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Rate limit hit, retry {attempt + 1}/{self.retry_attempts} after {wait_time}s",
                    extra={"attempt": attempt + 1, "wait_time": wait_time}
                )

                if attempt < self.retry_attempts - 1:
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(
                        f"Rate limit exceeded after {self.retry_attempts} attempts",
                        retry_after=wait_time
                    ) from e

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"API error, retrying: {e}")
                    time.sleep(1)
                else:
                    raise


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Client
# ═══════════════════════════════════════════════════════════════════════════════

class OpenAIClient(BaseLLMClient):
    """
    Client pour OpenAI GPT.

    MODÈLES SUPPORTÉS:
    - gpt-4o: Plus rapide, moins cher que GPT-4 Turbo
    - gpt-4o-mini: 15x moins cher, bon pour dev/tests
    - gpt-4-turbo: Ancienne version (plus cher)
    - gpt-3.5-turbo: Ancien, moins cher (pas recommandé pour math)

    Example:
        >>> config = load_config()
        >>> client = OpenAIClient(config)
        >>> response = client.generate("Qu'est-ce qu'une dérivée ?")
        >>> print(response.content)
    """

    def __init__(self, config: object, cost_tracker: Optional[CostTracker] = None):
        """
        Args:
            config: Objet Config
            cost_tracker: CostTracker

        Raises:
            MissingAPIKeyError: Si OPENAI_API_KEY manquante
        """
        super().__init__(config, cost_tracker)

        # Vérifier clé API
        if not config.api_keys.openai:
            raise MissingAPIKeyError("OpenAI")

        # Initialiser client
        self.client = OpenAI(
            api_key=config.api_keys.openai,
            timeout=self.timeout
        )

        self.model = config.llm.model

        logger.info(
            f"✓ OpenAI client initialized (manual Langfuse tracing)",
            extra={"model": self.model}
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        langfuse_span: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Génère une réponse avec OpenAI.

        Args:
            prompt: Prompt utilisateur
            system: Prompt système (optionnel)
            temperature: Override temperature
            max_tokens: Override max_tokens
            langfuse_span: Span Langfuse parent pour tracing (optionnel)
            **kwargs: Arguments additionnels pour API

        Returns:
            LLMResponse

        Example:
            >>> response = client.generate(
            ...     prompt="Expliquer les dérivées",
            ...     system="Tu es un prof de maths"
            ... )
        """
        # Messages
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        # Créer generation span Langfuse si disponible
        # Priorité: paramètre langfuse_span, puis context variable
        parent_span = langfuse_span or get_current_langfuse_span()
        generation_span = None

        if parent_span:
            try:
                generation_span = parent_span.generation(
                    name="openai_call",
                    model=self.model,
                    input=messages
                )
                logger.debug("Langfuse generation span created for OpenAI call")
            except Exception as e:
                logger.warning(f"Failed to create Langfuse generation span: {e}")

        # Paramètres
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Appel API avec retry
        def _call_api():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                **kwargs
            )

        try:
            with log_performance(logger, f"openai_api_call_{self.model}"):
                start_time = time.time()

                completion = self._retry_with_backoff(_call_api)

                latency = time.time() - start_time

            # Extraire résultat
            content = completion.choices[0].message.content

            # Tokens
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            total_tokens = completion.usage.total_tokens

            # Coût
            cost = self.cost_tracker.track_llm_call(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate"
            )

            # Finaliser generation span Langfuse
            if generation_span:
                try:
                    generation_span.end(
                        output=content,
                        usage={
                            "promptTokens": input_tokens,
                            "completionTokens": output_tokens,
                            "totalTokens": total_tokens
                        }
                    )
                    logger.debug("Langfuse generation span finalized")
                except Exception as e:
                    logger.warning(f"Failed to finalize Langfuse generation span: {e}")

            # Réponse
            response = LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "finish_reason": completion.choices[0].finish_reason,
                    "model_used": completion.model  # Peut différer de self.model
                }
            )

            logger.info(
                f"✓ OpenAI generation complete",
                extra={
                    "model": self.model,
                    "tokens": total_tokens,
                    "cost": cost,
                    "latency": latency
                }
            )

            return response

        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise LLMAPIError(
                f"OpenAI API call failed: {e}",
                provider="openai",
                status_code=getattr(e, 'status_code', None)
            ) from e


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic Client
# ═══════════════════════════════════════════════════════════════════════════════

class AnthropicClient(BaseLLMClient):
    """
    Client pour Anthropic Claude.

    MODÈLES SUPPORTÉS:
    - claude-sonnet-4: Équilibre qualité/prix (RECOMMANDÉ pour math)
    - claude-opus-4: Plus performant mais 5x plus cher
    - claude-haiku-4: Rapide et pas cher (simple tasks)

    POURQUOI CLAUDE POUR MATH:
    ✓ Meilleur raisonnement mathématique
    ✓ Préservation LaTeX plus fiable
    ✓ Contexte 200k tokens (vs 128k GPT)
    ✓ Moins d'hallucinations sur contenu technique

    Example:
        >>> config = load_config()
        >>> client = AnthropicClient(config)
        >>> response = client.generate("Démontrer que d(x²)/dx = 2x")
    """

    def __init__(self, config: object, cost_tracker: Optional[CostTracker] = None):
        """
        Args:
            config: Objet Config
            cost_tracker: CostTracker

        Raises:
            MissingAPIKeyError: Si ANTHROPIC_API_KEY manquante
        """
        super().__init__(config, cost_tracker)

        # Vérifier clé API
        if not config.api_keys.anthropic:
            raise MissingAPIKeyError("Anthropic")

        # Initialiser client
        self.client = Anthropic(
            api_key=config.api_keys.anthropic,
            timeout=self.timeout
        )

        self.model = config.llm.model

        logger.info(
            f"✓ Anthropic client initialized",
            extra={"model": self.model}
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Génère une réponse avec Claude.

        Args:
            prompt: Prompt utilisateur
            system: Prompt système (optionnel)
            temperature: Override temperature
            max_tokens: Override max_tokens
            **kwargs: Arguments additionnels pour API

        Returns:
            LLMResponse
        """
        # Paramètres
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Appel API avec retry
        def _call_api():
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system if system else "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )

        try:
            with log_performance(logger, f"anthropic_api_call_{self.model}"):
                start_time = time.time()

                message = self._retry_with_backoff(_call_api)

                latency = time.time() - start_time

            # Extraire résultat
            content = message.content[0].text

            # Tokens
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Coût
            cost = self.cost_tracker.track_llm_call(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate"
            )

            # Réponse
            response = LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "stop_reason": message.stop_reason,
                    "model_used": message.model
                }
            )

            logger.info(
                f"✓ Anthropic generation complete",
                extra={
                    "model": self.model,
                    "tokens": total_tokens,
                    "cost": cost,
                    "latency": latency
                }
            )

            return response

        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise LLMAPIError(
                f"Anthropic API call failed: {e}",
                provider="anthropic",
                status_code=getattr(e, 'status_code', None)
            ) from e


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama Client (Local)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaClient(BaseLLMClient):
    """
    Client pour modèles locaux via Ollama.

    MODÈLES SUPPORTÉS:
    - mistral:7b: Bon général, français OK
    - llama3:70b: Meilleur mais nécessite GPU puissant
    - codellama:7b: Spécialisé code (pas idéal pour math)

    INSTALLATION:
    1. Installer Ollama: https://ollama.ai/download
    2. Pull modèle: ollama pull mistral:7b
    3. Vérifier: ollama list

    LIMITATIONS:
    - Moins performant que Claude/GPT pour math avancé
    - Plus lent sur CPU
    - Pas de tracking tokens précis

    Example:
        >>> config = load_config()
        >>> client = OllamaClient(config)
        >>> response = client.generate("Expliquer les limites")
    """

    def __init__(self, config: object, cost_tracker: Optional[CostTracker] = None):
        """
        Args:
            config: Objet Config
            cost_tracker: CostTracker (coût = 0 pour local)
        """
        super().__init__(config, cost_tracker)

        # URL Ollama
        self.base_url = config.llm.ollama_base_url or "http://localhost:11434"
        self.model = config.llm.fallback_model or "mistral:7b"

        # Client HTTP simple (requests ou httpx)
        import requests
        self.session = requests.Session()

        logger.info(
            f"✓ Ollama client initialized",
            extra={"model": self.model, "base_url": self.base_url}
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Génère une réponse avec Ollama.

        Args:
            prompt: Prompt utilisateur
            system: Prompt système (optionnel)
            temperature: Override temperature
            max_tokens: Override max_tokens (num_predict dans Ollama)
            **kwargs: Arguments additionnels

        Returns:
            LLMResponse
        """
        # Construction du prompt complet
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"

        # Paramètres
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Payload Ollama
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "top_p": self.top_p
            }
        }

        try:
            with log_performance(logger, f"ollama_api_call_{self.model}"):
                start_time = time.time()

                # Appel API Ollama
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()

                latency = time.time() - start_time

            # Parser réponse
            data = response.json()
            content = data.get("response", "")

            # Tokens (approximation car Ollama ne fournit pas toujours)
            input_tokens = len(full_prompt) // 4  # Approximation
            output_tokens = len(content) // 4
            total_tokens = input_tokens + output_tokens

            # Coût = 0 (local)
            cost = 0.0

            # Réponse
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "eval_count": data.get("eval_count", output_tokens),
                    "eval_duration": data.get("eval_duration", 0)
                }
            )

            logger.info(
                f"✓ Ollama generation complete",
                extra={
                    "model": self.model,
                    "latency": latency,
                    "tokens_approx": total_tokens
                }
            )

            return llm_response

        except Exception as e:
            logger.error(f"Ollama API error: {e}", exc_info=True)
            raise LLMAPIError(
                f"Ollama API call failed: {e}. Is Ollama running?",
                provider="ollama"
            ) from e


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════════

def get_llm_client(
    config: object,
    cost_tracker: Optional[CostTracker] = None,
    force_provider: Optional[str] = None
) -> BaseLLMClient:
    """
    Factory pour créer le bon client LLM selon la configuration.

    Args:
        config: Objet Config
        cost_tracker: CostTracker (optionnel)
        force_provider: Forcer un provider ("openai", "anthropic", "local")

    Returns:
        Client LLM approprié

    Raises:
        ValueError: Si provider inconnu

    Example:
        >>> config = load_config()
        >>> client = get_llm_client(config)
        >>> response = client.generate("Test")
    """
    provider = force_provider or config.llm.provider

    if provider == "openai":
        return OpenAIClient(config, cost_tracker)

    elif provider == "anthropic":
        return AnthropicClient(config, cost_tracker)

    elif provider == "local":
        return OllamaClient(config, cost_tracker)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Use 'openai', 'anthropic', or 'local'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# CHOIX DU PROVIDER:
#
# OpenAI (GPT):
#   ✓ Rapide (GPT-4o)
#   ✓ Pas cher (GPT-4o-mini pour tests)
#   ✓ API stable
#   ✗ Moins bon en math que Claude
#   Use: Dev/tests, budget limité
#
# Anthropic (Claude):
#   ✓ Meilleur raisonnement mathématique
#   ✓ Contexte 200k tokens (vs 128k GPT)
#   ✓ Préservation LaTeX fiable
#   ✗ Un peu plus cher
#   Use: Production, math avancé
#
# Ollama (Local):
#   ✓ Gratuit
#   ✓ Privé (données restent locales)
#   ✓ Pas de rate limits
#   ✗ Moins performant
#   ✗ Nécessite GPU pour vitesse
#   Use: Dev offline, privacy, tests

#
# RETRY STRATEGY:
# - Exponential backoff: 1s, 2s, 4s
# - Retry sur RateLimitError
# - Max retry_attempts (config, défaut 3)
#
# COST TRACKING:
# - Automatique via cost_tracker
# - Logs à chaque appel
# - Alertes si budget dépassé
#
# TOKENS:
# - OpenAI: Précis (fourni par API)
# - Anthropic: Précis (fourni par API)
# - Ollama: Approximation (len / 4)
#
# FALLBACK:
# - Si API primary down → utiliser fallback_to_local
# - Implémentation dans les agents (pas ici)
#
# EXTENSIONS:
# - Support streaming (pour UI temps réel)
# - Function calling (OpenAI) / Tool use (Anthropic)
# - Fine-tuning personnalisé
# - Cache de réponses (éviter appels répétés)
#
# DEBUGGING:
# ```python
# # Tester
# from src.llm.closed_models import get_llm_client
# from src.utils.config_loader import load_config
#
# config = load_config()
# client = get_llm_client(config)
# response = client.generate("Test: 1+1=?")
# print(f"Réponse: {response.content}")
# print(f"Coût: ${response.cost:.4f}")
# print(f"Tokens: {response.total_tokens}")
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
