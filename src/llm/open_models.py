"""
Gestion des modèles ouverts (embeddings et modèles locaux).

Ce module centralise la configuration et le chargement des modèles open source.
Les embeddings sont déjà gérés par src/vectorization/embedder.py - ce module
fournit des utilitaires supplémentaires et une interface unifiée.

Usage:
    from src.llm.open_models import list_available_models, get_model_info

    models = list_available_models()
    info = get_model_info("all-MiniLM-L6-v2")
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Catalogue de modèles ouverts recommandés
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """Informations sur un modèle ouvert."""
    name: str
    type: str  # "embedding", "llm"
    dimensions: Optional[int]  # Pour embeddings
    size_mb: int
    languages: List[str]
    description: str
    recommended_for: str
    huggingface_id: str


# Catalogue de modèles d'embeddings recommandés
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": ModelInfo(
        name="all-MiniLM-L6-v2",
        type="embedding",
        dimensions=384,
        size_mb=80,
        languages=["en", "fr (limité)"],
        description="Modèle léger et rapide, bon équilibre qualité/vitesse",
        recommended_for="Usage général, MVP, développement",
        huggingface_id="sentence-transformers/all-MiniLM-L6-v2"
    ),

    "e5-large-v2": ModelInfo(
        name="e5-large-v2",
        type="embedding",
        dimensions=1024,
        size_mb=1200,
        languages=["en"],
        description="Meilleure qualité, plus lent",
        recommended_for="Production, qualité importante",
        huggingface_id="intfloat/e5-large-v2"
    ),

    "bge-large-en-v1.5": ModelInfo(
        name="bge-large-en-v1.5",
        type="embedding",
        dimensions=1024,
        size_mb=1300,
        languages=["en"],
        description="State-of-the-art, meilleures performances benchmarks",
        recommended_for="Production, recherche précise",
        huggingface_id="BAAI/bge-large-en-v1.5"
    ),

    "paraphrase-multilingual-mpnet": ModelInfo(
        name="paraphrase-multilingual-mpnet-base-v2",
        type="embedding",
        dimensions=768,
        size_mb=1000,
        languages=["50+ langues", "fr", "en"],
        description="Support multilingue, bon pour français+anglais",
        recommended_for="Documents multilingues",
        huggingface_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def list_available_models(model_type: Optional[str] = None) -> List[ModelInfo]:
    """
    Liste les modèles ouverts disponibles.

    Args:
        model_type: Filtrer par type ("embedding", "llm", None=tous)

    Returns:
        Liste de ModelInfo

    Example:
        >>> models = list_available_models("embedding")
        >>> for model in models:
        ...     print(f"{model.name}: {model.description}")
    """
    models = list(EMBEDDING_MODELS.values())

    if model_type:
        models = [m for m in models if m.type == model_type]

    return models


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    Récupère les infos d'un modèle.

    Args:
        model_name: Nom du modèle

    Returns:
        ModelInfo ou None si non trouvé

    Example:
        >>> info = get_model_info("all-MiniLM-L6-v2")
        >>> print(f"Dimensions: {info.dimensions}")
        Dimensions: 384
    """
    # Chercher par nom court
    if model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]

    # Chercher par HuggingFace ID
    for model_info in EMBEDDING_MODELS.values():
        if model_info.huggingface_id == model_name:
            return model_info

    return None


def recommend_model(
    use_case: str,
    budget: str = "medium"
) -> ModelInfo:
    """
    Recommande un modèle selon le cas d'usage.

    Args:
        use_case: "dev", "production", "multilingual"
        budget: "low", "medium", "high" (RAM/compute)

    Returns:
        ModelInfo recommandé

    Example:
        >>> model = recommend_model("dev", budget="low")
        >>> print(f"Recommandation: {model.name}")
        Recommandation: all-MiniLM-L6-v2
    """
    if use_case == "dev" or budget == "low":
        return EMBEDDING_MODELS["all-MiniLM-L6-v2"]

    elif use_case == "multilingual":
        return EMBEDDING_MODELS["paraphrase-multilingual-mpnet"]

    elif use_case == "production":
        if budget == "high":
            return EMBEDDING_MODELS["bge-large-en-v1.5"]
        else:
            return EMBEDDING_MODELS["e5-large-v2"]

    else:
        # Défaut
        return EMBEDDING_MODELS["all-MiniLM-L6-v2"]


def compare_models_table() -> str:
    """
    Génère un tableau comparatif des modèles.

    Returns:
        String formatée avec tableau

    Example:
        >>> print(compare_models_table())
    """
    models = list_available_models("embedding")

    lines = [
        "╔═══════════════════════════════════════════════════════════════════════════════╗",
        "║ MODÈLES D'EMBEDDINGS RECOMMANDÉS                                              ║",
        "╠═══════════════════════════════════════════════════════════════════════════════╣",
        "║ Modèle                  │ Dim  │ Taille │ Langues │ Recommandé pour          ║",
        "╠═════════════════════════╪══════╪════════╪═════════╪══════════════════════════╣",
    ]

    for model in models:
        # Tronquer si trop long
        name = model.name[:23]
        langs = ", ".join(model.languages[:2])[:7]
        rec = model.recommended_for[:22]

        line = (
            f"║ {name:23} │ {model.dimensions:4} │ {model.size_mb:4}MB │ "
            f"{langs:7} │ {rec:22} ║"
        )
        lines.append(line)

    lines.append(
        "╚═════════════════════════╧══════╧════════╧═════════╧══════════════════════════╝"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
# - Embeddings: Gérés par src/vectorization/embedder.py
# - Ce module: Catalogue et utilitaires
# - LLM locaux: Gérés par src/llm/closed_models.py (OllamaClient)
#
# MODÈLES RECOMMANDÉS PAR CAS D'USAGE:
#
# Développement/MVP:
#   → all-MiniLM-L6-v2 (rapide, léger, suffisant)
#
# Production (anglais):
#   → bge-large-en-v1.5 (meilleur)
#   → e5-large-v2 (bon compromis)
#
# Production (multilingue):
#   → paraphrase-multilingual-mpnet-base-v2
#
# Budget très limité (CPU faible):
#   → all-MiniLM-L6-v2 (seul choix viable)
#
# ALTERNATIVES:
# - OpenAI ada-002: Payant, très bon (0.0001$/1k tokens)
# - Cohere embeddings: Payant, bon
# - Instructor: Open source, instruction-based
#
# FINE-TUNING:
# Pour améliorer sur corpus mathématique spécifique:
# 1. Collecter paires (question, document_pertinent)
# 2. Fine-tuner avec sentence-transformers
# 3. Évaluer sur test set
# 4. Remplacer modèle dans config.yaml
#
# BENCHMARK:
# Utiliser MTEB (Massive Text Embedding Benchmark):
# https://huggingface.co/spaces/mteb/leaderboard
#
# ═══════════════════════════════════════════════════════════════════════════════
