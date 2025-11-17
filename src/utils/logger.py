"""
Configuration et utilitaires de logging structuré.

Ce module configure le système de logging pour tout le projet,
avec support de:
- Logs console colorés
- Logs fichiers avec rotation
- Format JSON structuré
- Logs par module
- Contexte enrichi

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_id": "123", "pages": 10})
"""

import logging
import logging.config
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# Optionnel: colorlog pour logs colorés (graceful degradation si absent)
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration du logging
# ═══════════════════════════════════════════════════════════════════════════════

_logging_configured = False


def setup_logging(
    config_path: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False
) -> None:
    """
    Configure le système de logging global.

    Args:
        config_path: Chemin vers logging_config.yaml (optionnel)
        log_level: Niveau de log par défaut (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Activer logs fichiers
        log_to_console: Activer logs console
        json_format: Utiliser format JSON (sinon format texte lisible)

    Note:
        Appelé automatiquement au premier appel de get_logger()
        Peut être appelé manuellement pour reconfigurer
    """
    global _logging_configured

    # Créer répertoire logs si nécessaire
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configuration programmatique (si pas de config.yaml ou pour override)
    handlers = []
    formatters = {}

    # Formatter console (avec couleurs si disponible)
    if log_to_console:
        if HAS_COLORLOG and not json_format:
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # Formatter fichier
    if log_to_file:
        if json_format:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        # Handler pour tous les logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

        # Handler pour erreurs uniquement
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)

    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Retirer handlers existants pour éviter doublons
    root_logger.handlers.clear()

    # Ajouter nos handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Réduire verbosité de bibliothèques tierces
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)

    _logging_configured = True


class JsonFormatter(logging.Formatter):
    """
    Formatter personnalisé pour logs en JSON structuré.

    Produit des logs parsables automatiquement pour ELK, Datadog, etc.

    Example output:
        {
          "timestamp": "2024-11-16T10:30:45.123Z",
          "level": "INFO",
          "module": "agents.retriever",
          "function": "search",
          "line": 42,
          "message": "Found 5 documents",
          "doc_count": 5,
          "query": "dérivée"
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un record de log en JSON.

        Args:
            record: Record de log à formater

        Returns:
            String JSON
        """
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "module": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Ajouter champs extra si présents
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "thread", "threadName", "exc_info", "exc_text", "stack_info"
                ]:
                    log_data[key] = value

        # Ajouter exception si présente
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger configuré pour un module.

    Args:
        name: Nom du logger (généralement __name__ du module)

    Returns:
        Logger configuré

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to process", extra={"file": "doc.pdf"})
    """
    global _logging_configured

    # Configuration automatique si pas encore fait
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Contextes de logging
# ═══════════════════════════════════════════════════════════════════════════════

class LogContext:
    """
    Context manager pour enrichir temporairement les logs avec du contexte.

    Permet d'ajouter automatiquement des informations de contexte à tous les logs
    dans un bloc de code.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(user_id="123", session_id="abc"):
        ...     logger.info("User logged in")  # Inclura user_id et session_id
        ...     process_user_data()
    """

    def __init__(self, **context):
        """
        Args:
            **context: Champs de contexte à ajouter aux logs
        """
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Entre dans le contexte."""
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sort du contexte."""
        logging.setLogRecordFactory(self.old_factory)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilitaires de logging
# ═══════════════════════════════════════════════════════════════════════════════

def log_function_call(logger: logging.Logger):
    """
    Décorateur pour logger automatiquement les appels de fonction.

    Args:
        logger: Logger à utiliser

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def process_document(doc_id: str, pages: int):
        ...     # ...
        ...     pass
        >>> process_document("123", 10)
        # Log: "Calling process_document(doc_id='123', pages=10)"
        # Log: "Finished process_document in 0.5s"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            # Log entrée
            args_str = ", ".join(repr(a) for a in args)
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))

            logger.debug(f"Calling {func.__name__}({all_args})")

            # Exécution
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log succès
                logger.debug(
                    f"Finished {func.__name__} in {duration:.3f}s",
                    extra={"duration": duration, "function": func.__name__}
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Log erreur
                logger.error(
                    f"Error in {func.__name__} after {duration:.3f}s: {e}",
                    extra={"duration": duration, "function": func.__name__},
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_performance(logger: logging.Logger, operation: str):
    """
    Context manager pour mesurer et logger la performance d'une opération.

    Args:
        logger: Logger à utiliser
        operation: Nom de l'opération

    Example:
        >>> logger = get_logger(__name__)
        >>> with log_performance(logger, "vector_search"):
        ...     results = search_vectors(query)
        # Log: "vector_search completed in 0.034s"
    """
    import time

    class PerformanceContext:
        def __enter__(self):
            self.start_time = time.time()
            logger.debug(f"{operation} started")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time

            if exc_type is None:
                logger.info(
                    f"{operation} completed in {duration:.3f}s",
                    extra={"operation": operation, "duration": duration}
                )
            else:
                logger.error(
                    f"{operation} failed after {duration:.3f}s",
                    extra={"operation": operation, "duration": duration},
                    exc_info=True
                )

    return PerformanceContext()


def parse_json_logs(log_file: Path) -> list:
    """
    Parse un fichier de logs JSON.

    Args:
        log_file: Chemin vers le fichier de logs

    Returns:
        Liste de dictionnaires (un par ligne de log)

    Example:
        >>> logs = parse_json_logs(Path("data/logs/app.log"))
        >>> errors = [log for log in logs if log["level"] == "ERROR"]
        >>> print(f"Found {len(errors)} errors")
    """
    logs = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                # Ligne non-JSON, ignorer
                continue

    return logs


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
# - setup_logging() configure le système global (appelé automatiquement)
# - get_logger() retourne des loggers configurés par module
# - JsonFormatter pour logs structurés (parsables automatiquement)
# - LogContext pour enrichir temporairement avec du contexte
#
# NIVEAUX DE LOG (du moins au plus sévère):
# - DEBUG: Détails pour débogage (variables, états)
# - INFO: Informations générales (étapes, succès)
# - WARNING: Avertissements (comportement inattendu mais géré)
# - ERROR: Erreurs (échecs d'opérations)
# - CRITICAL: Erreurs critiques (système inopérable)
#
# BONNES PRATIQUES:
# 1. Toujours utiliser get_logger(__name__) au début du module
# 2. Utiliser extra={} pour ajouter du contexte structuré
# 3. Logger les exceptions avec exc_info=True
# 4. Utiliser log_performance() pour opérations critiques
# 5. Format JSON pour production, texte pour développement
#
# ROTATION DES FICHIERS:
# - app.log: Tous les logs, max 10MB, 5 backups (50MB total)
# - errors.log: Erreurs uniquement, max 10MB, 10 backups (100MB total)
# - Rotation automatique quand taille max atteinte
#
# PARSING LOGS JSON:
# Avec jq (CLI):
#   cat data/logs/app.log | jq 'select(.level=="ERROR")'
#   cat data/logs/app.log | jq '.message'
#
# Avec Python:
#   logs = parse_json_logs(Path("data/logs/app.log"))
#   errors = [log for log in logs if log["level"] == "ERROR"]
#
# COLORLOG (optionnel):
# - Améliore lisibilité console
# - Graceful degradation si non installé
# - pip install colorlog
#
# VERBOSITÉ BIBLIOTHÈQUES TIERCES:
# - Automatiquement réduite pour transformers, httpx, etc.
# - Évite de polluer les logs avec détails internes
#
# EXEMPLE COMPLET:
# ```python
# from src.utils.logger import get_logger, LogContext, log_performance
#
# logger = get_logger(__name__)
#
# def process_document(doc_id: str):
#     with LogContext(doc_id=doc_id):
#         logger.info("Processing started")
#
#         with log_performance(logger, "extraction"):
#             text = extract_pdf(doc_id)
#
#         logger.info("Processing completed", extra={"chars": len(text)})
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
