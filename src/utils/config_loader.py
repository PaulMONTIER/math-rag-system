"""
Chargement et validation de la configuration système.

Ce module centralise le chargement de la configuration depuis:
- config/config.yaml (configuration principale)
- .env (variables d'environnement et clés API)
- Arguments CLI (overrides optionnels)

Usage:
    from src.utils.config_loader import load_config

    config = load_config()
    model_name = config.llm.model
    api_key = config.api_keys.openai
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError, MissingAPIKeyError


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses pour configuration typée
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoogleDriveConfig:
    """Configuration Google Drive API."""
    folder_id: str
    service_account_file: Optional[str]
    download_path: str
    auto_update: bool
    update_interval_hours: int


@dataclass
class PDFExtractionConfig:
    """Configuration extraction PDF."""
    library: str
    preserve_latex: bool
    extract_images: bool
    ocr_enabled: bool
    languages: list


@dataclass
class ChunkingConfig:
    """Configuration découpage de texte."""
    chunk_size: int
    chunk_overlap: int
    respect_formula_boundaries: bool
    respect_sentence_boundaries: bool
    min_chunk_size: int


@dataclass
class EmbeddingsConfig:
    """Configuration modèle d'embeddings."""
    model: str
    batch_size: int
    device: str
    cache_embeddings: bool
    cache_path: str


@dataclass
class VectorStoreConfig:
    """Configuration base vectorielle."""
    type: str
    index_type: str
    persist_path: str
    similarity_metric: str


@dataclass
class RetrievalConfig:
    """Configuration recherche RAG."""
    top_k: int
    similarity_threshold: float
    use_reranking: bool
    reranking_model: Optional[str] = None
    max_context_tokens: int = 3000


@dataclass
class LLMConfig:
    """Configuration LLM (modèle fermé)."""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    timeout: int
    retry_attempts: int
    fallback_to_local: bool
    fallback_model: Optional[str]
    ollama_base_url: Optional[str] = None


@dataclass
class AgentsConfig:
    """Configuration des agents."""
    classifier: Dict[str, Any]
    retriever: Dict[str, Any]
    generator: Dict[str, Any]
    verifier: Dict[str, Any]
    validator: Dict[str, Any]


@dataclass
class WorkflowConfig:
    """Configuration workflow LangGraph."""
    enable_logging: bool
    log_level: str
    visualize_graph: bool
    save_state: bool
    save_state_path: Optional[str] = None


@dataclass
class InterfaceConfig:
    """Configuration interface web."""
    framework: str
    host: str
    port: int
    enable_metrics: bool
    enable_feedback: bool
    session_timeout_minutes: int
    theme: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostsConfig:
    """Configuration tracking des coûts."""
    track_costs: bool
    max_cost_per_session: float
    alert_threshold: float
    pricing: Dict[str, Dict[str, float]]


@dataclass
class LoggingConfig:
    """Configuration logging."""
    level: str
    console: bool
    file: bool
    file_path: str
    rotation: str
    retention_days: int
    format: str
    modules: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestingConfig:
    """Configuration tests."""
    test_questions_file: str
    run_benchmarks: bool
    compare_models: bool
    models_to_compare: list = field(default_factory=list)


@dataclass
class PedagogyConfig:
    """Configuration pédagogie."""
    student_levels: list
    adapt_response_level: bool
    provide_examples: bool
    cite_sources: bool
    suggest_prerequisites: bool
    suggest_exercises: bool = False


@dataclass
class MonitoringConfig:
    """Configuration monitoring (Langfuse)."""
    enabled: bool
    trace_all: bool
    trace_errors_only: bool


@dataclass
class APIKeys:
    """Clés API (depuis .env)."""
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    google_credentials: Optional[str] = None
    langfuse_public: Optional[str] = None
    langfuse_secret: Optional[str] = None
    langfuse_base_url: Optional[str] = None


@dataclass
class Config:
    """Configuration complète du système."""
    # Modules
    google_drive: GoogleDriveConfig
    pdf_extraction: PDFExtractionConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    agents: AgentsConfig
    workflow: WorkflowConfig
    interface: InterfaceConfig
    costs: CostsConfig
    logging: LoggingConfig
    testing: TestingConfig
    pedagogy: PedagogyConfig
    monitoring: MonitoringConfig

    # API Keys
    api_keys: APIKeys

    # Métadonnées
    environment: str = "development"
    debug: bool = False
    project_root: Path = field(default_factory=lambda: Path.cwd())


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions de chargement
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_root() -> Path:
    """
    Détecte la racine du projet.

    Cherche le dossier contenant config/config.yaml en remontant l'arborescence.

    Returns:
        Path: Chemin absolu vers la racine du projet

    Raises:
        ConfigurationError: Si la racine n'est pas trouvée
    """
    current = Path.cwd()

    # Remonter jusqu'à trouver config/config.yaml
    for _ in range(5):  # Max 5 niveaux
        config_file = current / "config" / "config.yaml"
        if config_file.exists():
            return current
        current = current.parent

    raise ConfigurationError(
        "Impossible de trouver la racine du projet (config/config.yaml introuvable). "
        "Assurez-vous d'exécuter depuis le répertoire du projet."
    )


def load_env_variables(project_root: Path) -> None:
    """
    Charge les variables d'environnement depuis .env.

    Args:
        project_root: Racine du projet

    Note:
        Si .env n'existe pas, affiche un warning mais continue
        (permet de fonctionner avec variables d'env système)
    """
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Warning mais pas d'erreur (peut utiliser variables système)
        import warnings
        warnings.warn(
            f".env file not found at {env_file}. "
            "Using system environment variables. "
            "Create .env from .env.example if needed.",
            UserWarning
        )


def load_yaml_config(project_root: Path) -> Dict[str, Any]:
    """
    Charge config/config.yaml.

    Args:
        project_root: Racine du projet

    Returns:
        Dict contenant la configuration

    Raises:
        ConfigurationError: Si le fichier est invalide ou manquant
    """
    config_file = project_root / "config" / "config.yaml"

    if not config_file.exists():
        raise ConfigurationError(
            f"Fichier de configuration introuvable: {config_file}"
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Erreur de parsing YAML dans {config_file}: {e}"
        )
    except Exception as e:
        raise ConfigurationError(
            f"Erreur lors de la lecture de {config_file}: {e}"
        )


def load_api_keys() -> APIKeys:
    """
    Charge les clés API depuis les variables d'environnement.

    Returns:
        APIKeys: Objet contenant toutes les clés API

    Note:
        Les clés manquantes sont None (pas d'erreur ici, la validation vient après)
    """
    return APIKeys(
        openai=os.getenv("OPENAI_API_KEY"),
        anthropic=os.getenv("ANTHROPIC_API_KEY"),
        google_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        langfuse_public=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    )


def auto_detect_device() -> str:
    """
    Détecte automatiquement le meilleur device pour PyTorch.

    Returns:
        str: "cuda", "mps" (Apple Silicon), ou "cpu"

    Note:
        Utilisé si config.embeddings.device = "auto"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def parse_config_dict(config_dict: Dict[str, Any], api_keys: APIKeys, project_root: Path) -> Config:
    """
    Parse le dictionnaire de configuration en objets typés.

    Args:
        config_dict: Configuration brute depuis YAML
        api_keys: Clés API chargées
        project_root: Racine du projet

    Returns:
        Config: Configuration complète et typée

    Raises:
        ConfigurationError: Si des champs obligatoires manquent
    """
    try:
        # Auto-détection du device si nécessaire
        embeddings_config = config_dict.get("embeddings", {})
        if embeddings_config.get("device") == "auto":
            embeddings_config["device"] = auto_detect_device()

        # Construction de l'objet Config
        config = Config(
            google_drive=GoogleDriveConfig(**config_dict["google_drive"]),
            pdf_extraction=PDFExtractionConfig(**config_dict["pdf_extraction"]),
            chunking=ChunkingConfig(**config_dict["chunking"]),
            embeddings=EmbeddingsConfig(**embeddings_config),
            vector_store=VectorStoreConfig(**config_dict["vector_store"]),
            retrieval=RetrievalConfig(**config_dict["retrieval"]),
            llm=LLMConfig(**config_dict["llm"]),
            agents=AgentsConfig(**config_dict["agents"]),
            workflow=WorkflowConfig(**config_dict["workflow"]),
            interface=InterfaceConfig(**config_dict["interface"]),
            costs=CostsConfig(**config_dict["costs"]),
            logging=LoggingConfig(**config_dict["logging"]),
            testing=TestingConfig(**config_dict["testing"]),
            pedagogy=PedagogyConfig(**config_dict["pedagogy"]),
            monitoring=MonitoringConfig(**config_dict.get("monitoring", {
                "enabled": False,
                "trace_all": True,
                "trace_errors_only": False
            })),
            api_keys=api_keys,
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            project_root=project_root
        )

        return config

    except KeyError as e:
        raise ConfigurationError(
            f"Champ de configuration manquant: {e}. "
            "Vérifiez config/config.yaml"
        )
    except TypeError as e:
        raise ConfigurationError(
            f"Type de configuration invalide: {e}. "
            "Vérifiez les valeurs dans config/config.yaml"
        )


def validate_config(config: Config) -> None:
    """
    Valide la configuration et vérifie les dépendances.

    Args:
        config: Configuration à valider

    Raises:
        ConfigurationError: Si la configuration est invalide
        MissingAPIKeyError: Si une clé API nécessaire manque

    Validations:
    - Clés API nécessaires présentes selon le provider LLM
    - Chemins de fichiers valides
    - Valeurs dans des ranges acceptables
    - Cohérence entre options
    """
    # Vérifier clés API selon provider
    if config.llm.provider == "anthropic":
        if not config.api_keys.anthropic:
            raise MissingAPIKeyError("Anthropic")
    elif config.llm.provider == "openai":
        if not config.api_keys.openai:
            raise MissingAPIKeyError("OpenAI")

    # Vérifier monitoring Langfuse si activé
    if config.monitoring.enabled:
        if not config.api_keys.langfuse_public or not config.api_keys.langfuse_secret:
            raise MissingAPIKeyError("Langfuse")

    # Vérifier valeurs dans ranges acceptables
    if config.chunking.chunk_size < 50:
        raise ConfigurationError("chunk_size doit être >= 50")

    if config.chunking.chunk_overlap >= config.chunking.chunk_size:
        raise ConfigurationError("chunk_overlap doit être < chunk_size")

    if not 0 <= config.llm.temperature <= 1:
        raise ConfigurationError("temperature doit être entre 0 et 1")

    if config.retrieval.top_k < 1:
        raise ConfigurationError("top_k doit être >= 1")

    # Vérifier cohérence re-ranking
    if config.retrieval.use_reranking and not config.retrieval.reranking_model:
        raise ConfigurationError(
            "use_reranking=true nécessite de spécifier reranking_model"
        )

    # Créer répertoires nécessaires s'ils n'existent pas
    directories = [
        config.google_drive.download_path,
        config.vector_store.persist_path,
        config.embeddings.cache_path if config.embeddings.cache_embeddings else None,
        Path(config.logging.file_path).parent,
        Path(config.workflow.save_state_path).parent if config.workflow.save_state else None,
    ]

    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Point d'entrée principal pour charger la configuration.

    Args:
        config_path: Chemin optionnel vers config.yaml (par défaut: auto-détection)

    Returns:
        Config: Configuration complète, validée et prête à l'emploi

    Raises:
        ConfigurationError: Si la configuration est invalide
        MissingAPIKeyError: Si des clés API nécessaires manquent

    Example:
        >>> config = load_config()
        >>> print(config.llm.model)
        gpt-4o
        >>> print(config.api_keys.openai)
        sk-...
    """
    # Détection de la racine du projet
    if config_path:
        project_root = config_path.parent.parent
    else:
        project_root = get_project_root()

    # Chargement .env
    load_env_variables(project_root)

    # Chargement config.yaml
    config_dict = load_yaml_config(project_root)

    # Chargement API keys
    api_keys = load_api_keys()

    # Parsing en objets typés
    config = parse_config_dict(config_dict, api_keys, project_root)

    # Validation
    validate_config(config)

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def get_config_value(config: Config, path: str, default: Any = None) -> Any:
    """
    Récupère une valeur de configuration via un chemin pointé.

    Args:
        config: Configuration
        path: Chemin pointé (ex: "llm.model", "retrieval.top_k")
        default: Valeur par défaut si non trouvée

    Returns:
        Valeur de configuration ou default

    Example:
        >>> config = load_config()
        >>> model = get_config_value(config, "llm.model")
        >>> top_k = get_config_value(config, "retrieval.top_k", 5)
    """
    parts = path.split(".")
    current = config

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return default

    return current


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
# 1. load_config() est le point d'entrée principal
# 2. Charge .env puis config.yaml
# 3. Parse en dataclasses typées (type safety!)
# 4. Valide la configuration
# 5. Crée les répertoires nécessaires
#
# AVANTAGES DES DATACLASSES:
# - Type hints automatiques (IDE autocomplete)
# - Validation au runtime
# - Documentation claire
# - Sérialisation facile
#
# ORDRE DE PRIORITÉ (du plus au moins prioritaire):
# 1. Variables d'environnement système
# 2. .env
# 3. config.yaml
# 4. Valeurs par défaut dans les dataclasses
#
# EXTENSION:
# Pour ajouter un nouveau paramètre:
# 1. Ajouter dans config.yaml
# 2. Ajouter dans la dataclass correspondante
# 3. (Optionnel) Ajouter validation dans validate_config()
#
# AUTO-DÉTECTION:
# - Device (CPU/CUDA/MPS): automatique si device="auto"
# - Project root: remonte jusqu'à trouver config/
#
# SÉCURITÉ:
# - .env jamais committé (dans .gitignore)
# - Clés API jamais loggées
# - Validation stricte pour éviter injections
#
# ═══════════════════════════════════════════════════════════════════════════════
