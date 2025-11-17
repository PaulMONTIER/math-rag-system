"""
Exceptions personnalisées pour le système RAG hybride.

Ce module définit toutes les exceptions spécifiques au système,
permettant une gestion d'erreurs granulaire et des messages clairs.

Usage:
    from src.utils.exceptions import PDFExtractionError

    try:
        extract_pdf(path)
    except PDFExtractionError as e:
        logger.error(f"Extraction failed: {e}")
"""

from typing import Optional


class RagSystemError(Exception):
    """
    Exception de base pour toutes les erreurs du système RAG.

    Toutes les exceptions custom héritent de celle-ci,
    permettant un catch global si nécessaire.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Args:
            message: Message d'erreur principal
            details: Dictionnaire de détails additionnels (optionnel)
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Représentation string avec détails si présents."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - Extraction de PDFs
# ═══════════════════════════════════════════════════════════════════════════════

class PDFExtractionError(RagSystemError):
    """
    Erreur lors de l'extraction de contenu depuis un PDF.

    Causes possibles:
    - Fichier PDF corrompu
    - Format non supporté
    - Erreur de parsing
    - Permissions insuffisantes

    Suggestion: Vérifier l'intégrité du PDF, essayer une autre bibliothèque
    """
    pass


class LatexParsingError(RagSystemError):
    """
    Erreur lors du parsing de formules LaTeX.

    Causes possibles:
    - Syntaxe LaTeX invalide
    - Commandes non supportées
    - Formule incomplète

    Suggestion: Valider la formule LaTeX manuellement
    """
    pass


class GoogleDriveError(RagSystemError):
    """
    Erreur lors de l'interaction avec Google Drive API.

    Causes possibles:
    - Credentials invalides ou expirés
    - Quota API dépassé
    - Dossier inexistant
    - Permissions insuffisantes

    Suggestion: Vérifier credentials, quotas API, permissions du dossier
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - Vectorisation
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingError(RagSystemError):
    """
    Erreur lors de la génération d'embeddings.

    Causes possibles:
    - Modèle d'embedding non chargé
    - Texte trop long
    - Erreur CUDA (GPU)
    - Mémoire insuffisante

    Suggestion: Vérifier que le modèle est bien chargé, réduire batch_size
    """
    pass


class VectorStoreError(RagSystemError):
    """
    Erreur lors des opérations sur la base vectorielle.

    Causes possibles:
    - Index FAISS corrompu
    - Dimensions incompatibles
    - Index non initialisé
    - Erreur de persistance

    Suggestion: Reconstruire l'index, vérifier dimensions des vecteurs
    """
    pass


class ChunkingError(RagSystemError):
    """
    Erreur lors du découpage du texte en chunks.

    Causes possibles:
    - Texte vide
    - Paramètres de chunking invalides
    - Formule LaTeX non fermée

    Suggestion: Vérifier paramètres chunk_size et chunk_overlap
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - LLM et APIs
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAPIError(RagSystemError):
    """
    Erreur lors de l'appel aux APIs de LLM (Claude, GPT, etc.).

    Causes possibles:
    - Clé API invalide ou expirée
    - Rate limit dépassé
    - Timeout
    - Quota dépassé
    - Erreur serveur (500)

    Suggestion: Vérifier clé API, attendre avant retry, vérifier quotas
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        """
        Args:
            message: Message d'erreur
            provider: Provider LLM (anthropic, openai, etc.)
            status_code: Code HTTP si applicable
            retry_after: Secondes avant de pouvoir retry (si rate limit)
        """
        details = {}
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(message, details)


class ModelNotFoundError(RagSystemError):
    """
    Erreur lorsqu'un modèle requis n'est pas trouvé.

    Causes possibles:
    - Modèle non téléchargé
    - Nom de modèle incorrect
    - Chemin invalide

    Suggestion: Vérifier que le modèle est bien installé (pip install, huggingface)
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - Workflow et Agents
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowError(RagSystemError):
    """
    Erreur lors de l'exécution du workflow LangGraph.

    Causes possibles:
    - État du workflow invalide
    - Transition impossible
    - Agent défaillant
    - Timeout

    Suggestion: Vérifier logs du workflow, état des agents
    """
    pass


class AgentError(RagSystemError):
    """
    Erreur lors de l'exécution d'un agent spécifique.

    Causes possibles:
    - Configuration agent invalide
    - Données d'entrée manquantes
    - Erreur de traitement

    Suggestion: Vérifier configuration de l'agent, logs détaillés
    """

    def __init__(self, message: str, agent_name: Optional[str] = None):
        """
        Args:
            message: Message d'erreur
            agent_name: Nom de l'agent en erreur
        """
        details = {"agent": agent_name} if agent_name else {}
        super().__init__(message, details)


class ValidationError(RagSystemError):
    """
    Erreur lors de la validation (humaine ou automatique).

    Causes possibles:
    - Réponse rejetée par le validateur
    - Hallucinations détectées
    - Qualité insuffisante

    Suggestion: Ajuster paramètres de vérification, améliorer prompts
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigurationError(RagSystemError):
    """
    Erreur de configuration du système.

    Causes possibles:
    - Fichier config.yaml invalide
    - Variables d'environnement manquantes
    - Valeurs de configuration invalides

    Suggestion: Vérifier config.yaml et .env, comparer avec .env.example
    """
    pass


class MissingAPIKeyError(ConfigurationError):
    """
    Clé API manquante dans la configuration.

    Causes possibles:
    - .env non créé ou incomplet
    - Variable d'environnement non définie

    Suggestion: Créer .env depuis .env.example et remplir les clés API
    """

    def __init__(self, api_name: str):
        """
        Args:
            api_name: Nom de l'API dont la clé est manquante
        """
        message = (
            f"Clé API manquante pour {api_name}. "
            f"Veuillez définir la variable d'environnement correspondante dans .env"
        )
        super().__init__(message, details={"api": api_name})


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions - Interface
# ═══════════════════════════════════════════════════════════════════════════════

class InterfaceError(RagSystemError):
    """
    Erreur dans l'interface web.

    Causes possibles:
    - Erreur de rendering
    - Session expirée
    - Données invalides du frontend

    Suggestion: Rafraîchir la page, vérifier logs Streamlit
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires pour gestion d'erreurs
# ═══════════════════════════════════════════════════════════════════════════════

def format_error_message(error: Exception) -> str:
    """
    Formate un message d'erreur de manière user-friendly.

    Args:
        error: Exception à formater

    Returns:
        Message d'erreur formaté avec suggestions

    Example:
        >>> try:
        ...     raise PDFExtractionError("Failed to parse PDF")
        ... except Exception as e:
        ...     print(format_error_message(e))
        ❌ Erreur d'extraction PDF: Failed to parse PDF
        💡 Suggestion: Vérifier l'intégrité du PDF, essayer une autre bibliothèque
    """
    error_class = error.__class__.__name__
    error_msg = str(error)

    # Mapping des types d'erreurs vers des emojis et suggestions
    error_info = {
        "PDFExtractionError": ("📄", "Vérifier l'intégrité du PDF"),
        "LatexParsingError": ("🔢", "Valider la syntaxe LaTeX"),
        "GoogleDriveError": ("☁️", "Vérifier credentials et permissions"),
        "EmbeddingError": ("🧮", "Vérifier modèle et mémoire"),
        "VectorStoreError": ("🗄️", "Reconstruire l'index vectoriel"),
        "LLMAPIError": ("🤖", "Vérifier clé API et quotas"),
        "WorkflowError": ("⚙️", "Consulter logs du workflow"),
        "ConfigurationError": ("⚙️", "Vérifier config.yaml et .env"),
    }

    emoji, suggestion = error_info.get(error_class, ("❌", "Consulter la documentation"))

    formatted = f"{emoji} {error_class}: {error_msg}"
    if suggestion:
        formatted += f"\n💡 Suggestion: {suggestion}"

    return formatted


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# HIÉRARCHIE DES EXCEPTIONS:
# RagSystemError (base)
#   ├── PDFExtractionError
#   ├── LatexParsingError
#   ├── GoogleDriveError
#   ├── EmbeddingError
#   ├── VectorStoreError
#   ├── ChunkingError
#   ├── LLMAPIError
#   ├── ModelNotFoundError
#   ├── WorkflowError
#   ├── AgentError
#   ├── ValidationError
#   ├── ConfigurationError
#   │   └── MissingAPIKeyError
#   └── InterfaceError
#
# USAGE RECOMMANDÉ:
# 1. Catch spécifique quand possible:
#    try:
#        extract_pdf()
#    except PDFExtractionError as e:
#        handle_pdf_error(e)
#
# 2. Catch générique pour fallback:
#    try:
#        run_workflow()
#    except RagSystemError as e:
#        logger.error(format_error_message(e))
#        fallback_response()
#
# 3. Toujours logger les exceptions:
#    except Exception as e:
#        logger.exception("Unexpected error", exc_info=True)
#        raise WorkflowError("Workflow failed") from e
#
# BONNES PRATIQUES:
# - Fournir des messages d'erreur clairs et actionnables
# - Inclure le contexte dans `details` (IDs, params, etc.)
# - Suggérer des solutions dans les docstrings
# - Logger avant de raise (sauf si re-raise)
# - Utiliser `raise ... from e` pour préserver la stack trace
#
# ═══════════════════════════════════════════════════════════════════════════════
