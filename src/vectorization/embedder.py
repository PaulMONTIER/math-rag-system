"""
Génération d'embeddings avec sentence-transformers (MODÈLE OUVERT).

Ce module transforme du texte en vecteurs denses pour recherche sémantique.
Utilise des modèles pré-entraînés de Hugging Face.

MODÈLE PAR DÉFAUT: all-MiniLM-L6-v2
- Dimension: 384
- Taille: ~80MB
- Rapide et efficace

Usage:
    from src.vectorization.embedder import Embedder

    embedder = Embedder(config)
    vectors = embedder.embed_texts(["Question sur dérivées", "Théorème"])
    print(f"Shape: {vectors.shape}")  # (2, 384)
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict
import hashlib
import pickle

from sentence_transformers import SentenceTransformer
import torch

from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import EmbeddingError, ModelNotFoundError

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedder
# ═══════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Générateur d'embeddings avec modèles sentence-transformers.

    MODÈLES RECOMMANDÉS:
    - all-MiniLM-L6-v2: Léger, rapide (384 dim, ~80MB) [DÉFAUT]
    - intfloat/e5-large-v2: Meilleur qualité (1024 dim, ~1.2GB)
    - BAAI/bge-large-en-v1.5: State-of-the-art (1024 dim, ~1.3GB)
    - paraphrase-multilingual-mpnet-base-v2: Multilingue (768 dim, ~1GB)

    Example:
        >>> config = load_config()
        >>> embedder = Embedder(config)
        >>> vectors = embedder.embed_texts(["texte 1", "texte 2"])
        >>> print(vectors.shape)
        (2, 384)
    """

    def __init__(self, config: Optional[object] = None):
        """
        Args:
            config: Objet Config avec embeddings settings
        """
        # Configuration
        if config and hasattr(config, 'embeddings'):
            self.model_name = config.embeddings.model
            self.batch_size = config.embeddings.batch_size
            self.device = config.embeddings.device
            self.cache_embeddings = config.embeddings.cache_embeddings
            self.cache_path = Path(config.embeddings.cache_path) if config.embeddings.cache_embeddings else None
        else:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.batch_size = 32
            self.device = "auto"
            self.cache_embeddings = False
            self.cache_path = None

        # Auto-détecter device si besoin
        if self.device == "auto":
            self.device = self._auto_detect_device()

        logger.info(
            f"Initializing Embedder with model: {self.model_name}",
            extra={"model": self.model_name, "device": self.device}
        )

        # Charger modèle
        try:
            self.model = self._load_model()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"✓ Embedder initialized successfully",
                extra={
                    "model": self.model_name,
                    "dimension": self.embedding_dim,
                    "device": self.device,
                    "batch_size": self.batch_size
                }
            )

        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to load embedding model {self.model_name}: {e}"
            ) from e

        # Cache
        if self.cache_embeddings and self.cache_path:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self._cache: Dict[str, np.ndarray] = {}
            self._load_cache()

    def _auto_detect_device(self) -> str:
        """
        Détecte automatiquement le meilleur device.

        Returns:
            "cuda", "mps" (Apple Silicon), ou "cpu"
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"✓ CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("✓ Apple Silicon (MPS) available, using GPU")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU detected)")

        return device

    def _load_model(self) -> SentenceTransformer:
        """
        Charge le modèle sentence-transformers.

        Returns:
            Modèle chargé

        Raises:
            ModelNotFoundError: Si modèle introuvable
        """
        try:
            # Charger modèle
            model = SentenceTransformer(self.model_name, device=self.device)

            return model

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")

            # Message d'aide
            if "not found" in str(e).lower() or "404" in str(e):
                raise ModelNotFoundError(
                    f"Model '{self.model_name}' not found. "
                    f"Check model name on https://huggingface.co/models"
                ) from e
            else:
                raise ModelNotFoundError(
                    f"Error loading model '{self.model_name}': {e}"
                ) from e

    def embed_text(self, text: str) -> np.ndarray:
        """
        Génère l'embedding d'un seul texte.

        Args:
            text: Texte à embedder

        Returns:
            Vecteur numpy de dimension (embedding_dim,)

        Example:
            >>> vec = embedder.embed_text("Qu'est-ce qu'une dérivée ?")
            >>> print(vec.shape)
            (384,)
        """
        if not text or not text.strip():
            raise EmbeddingError("Texte vide fourni pour embedding")

        # Vérifier cache
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug("✓ Embedding found in cache")
                return self._cache[cache_key]

        # Générer embedding
        with log_performance(logger, "embed_single_text"):
            try:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                # Mettre en cache
                if self.cache_embeddings:
                    self._cache[cache_key] = embedding

                return embedding

            except Exception as e:
                raise EmbeddingError(
                    f"Failed to generate embedding: {e}",
                    details={"text_length": len(text)}
                ) from e

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Génère les embeddings de plusieurs textes (batch processing).

        Args:
            texts: Liste de textes
            show_progress: Afficher barre de progression

        Returns:
            Matrice numpy de dimension (len(texts), embedding_dim)

        Example:
            >>> texts = ["texte 1", "texte 2", "texte 3"]
            >>> vectors = embedder.embed_texts(texts)
            >>> print(vectors.shape)
            (3, 384)
        """
        if not texts:
            raise EmbeddingError("Liste de textes vide")

        # Filtrer textes vides
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(
                f"{len(texts) - len(valid_texts)} textes vides ignorés",
                extra={"total": len(texts), "valid": len(valid_texts)}
            )

        if not valid_texts:
            raise EmbeddingError("Tous les textes sont vides")

        # Vérifier cache (pour tous)
        embeddings_list = []
        texts_to_embed = []
        text_indices = []

        if self.cache_embeddings:
            for i, text in enumerate(valid_texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings_list.append((i, self._cache[cache_key]))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)

            if embeddings_list:
                logger.debug(f"✓ {len(embeddings_list)} embeddings from cache")
        else:
            texts_to_embed = valid_texts
            text_indices = list(range(len(valid_texts)))

        # Générer embeddings manquants
        if texts_to_embed:
            with log_performance(logger, f"embed_batch_{len(texts_to_embed)}_texts"):
                try:
                    new_embeddings = self.model.encode(
                        texts_to_embed,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress
                    )

                    # Ajouter à la liste
                    for i, idx in enumerate(text_indices):
                        embeddings_list.append((idx, new_embeddings[i]))

                        # Mettre en cache
                        if self.cache_embeddings:
                            cache_key = self._get_cache_key(texts_to_embed[i])
                            self._cache[cache_key] = new_embeddings[i]

                except Exception as e:
                    raise EmbeddingError(
                        f"Failed to generate batch embeddings: {e}",
                        details={"batch_size": len(texts_to_embed)}
                    ) from e

        # Trier par index original
        embeddings_list.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in embeddings_list])

        logger.info(
            f"✓ Generated {len(embeddings)} embeddings",
            extra={
                "count": len(embeddings),
                "dimension": self.embedding_dim,
                "from_cache": len(valid_texts) - len(texts_to_embed) if self.cache_embeddings else 0
            }
        )

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """
        Génère une clé de cache pour un texte.

        Args:
            text: Texte

        Returns:
            Hash MD5 du texte
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_cache(self) -> None:
        """Charge le cache depuis le disque si présent."""
        if not self.cache_path:
            return

        cache_file = self.cache_path / f"embeddings_cache_{self.model_name.replace('/', '_')}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)

                logger.info(
                    f"✓ Loaded {len(self._cache)} embeddings from cache",
                    extra={"cache_file": str(cache_file)}
                )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def save_cache(self) -> None:
        """Sauvegarde le cache sur le disque."""
        if not self.cache_embeddings or not self.cache_path:
            return

        cache_file = self.cache_path / f"embeddings_cache_{self.model_name.replace('/', '_')}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)

            logger.info(
                f"✓ Saved {len(self._cache)} embeddings to cache",
                extra={"cache_file": str(cache_file)}
            )
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_embedding_dimension(self) -> int:
        """
        Retourne la dimension des embeddings.

        Returns:
            Dimension (ex: 384 pour all-MiniLM-L6-v2)
        """
        return self.embedding_dim

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calcule la similarité entre deux embeddings.

        Args:
            embedding1: Premier embedding
            embedding2: Second embedding
            metric: Métrique ("cosine", "l2", "dot")

        Returns:
            Score de similarité

        Example:
            >>> vec1 = embedder.embed_text("dérivée")
            >>> vec2 = embedder.embed_text("primitive")
            >>> similarity = embedder.compute_similarity(vec1, vec2)
            >>> print(f"Similarité: {similarity:.3f}")
        """
        if metric == "cosine":
            # Similarité cosinus : dot product des vecteurs normalisés
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return np.dot(embedding1, embedding2) / (norm1 * norm2)

        elif metric == "l2":
            # Distance L2 (Euclidienne) - inversée pour similarité
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1 / (1 + distance)  # Convertir distance en similarité

        elif metric == "dot":
            # Produit scalaire (dot product)
            return np.dot(embedding1, embedding2)

        else:
            raise ValueError(f"Unknown metric: {metric}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def compare_embedders(
    texts: List[str],
    models: List[str]
) -> Dict[str, Dict]:
    """
    Compare plusieurs modèles d'embeddings.

    Args:
        texts: Textes de test
        models: Liste de noms de modèles

    Returns:
        Dict avec résultats par modèle

    Example:
        >>> texts = ["test 1", "test 2"]
        >>> models = ["all-MiniLM-L6-v2", "e5-large-v2"]
        >>> results = compare_embedders(texts, models)
        >>> for model, stats in results.items():
        ...     print(f"{model}: {stats['time']:.2f}s, dim={stats['dimension']}")
    """
    import time

    results = {}

    for model_name in models:
        logger.info(f"Testing model: {model_name}")

        try:
            # Créer embedder temporaire
            temp_embedder = Embedder()
            temp_embedder.model_name = model_name
            temp_embedder.model = SentenceTransformer(model_name)
            temp_embedder.embedding_dim = temp_embedder.model.get_sentence_embedding_dimension()

            # Mesurer temps
            start = time.time()
            embeddings = temp_embedder.embed_texts(texts, show_progress=False)
            duration = time.time() - start

            results[model_name] = {
                "dimension": temp_embedder.embedding_dim,
                "time": duration,
                "time_per_text": duration / len(texts),
                "success": True
            }

        except Exception as e:
            results[model_name] = {
                "success": False,
                "error": str(e)
            }
            logger.error(f"Failed to test {model_name}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# MODÈLES SENTENCE-TRANSFORMERS:
# - all-MiniLM-L6-v2: DÉFAUT
#   * 384 dimensions
#   * ~80MB
#   * Rapide (~100 texts/sec CPU)
#   * Bon pour la plupart des cas
#
# - intfloat/e5-large-v2: QUALITÉ
#   * 1024 dimensions
#   * ~1.2GB
#   * Plus lent (~30 texts/sec CPU)
#   * Meilleur qualité embeddings
#
# - BAAI/bge-large-en-v1.5: STATE-OF-THE-ART
#   * 1024 dimensions
#   * ~1.3GB
#   * Plus lent (~25 texts/sec CPU)
#   * Meilleures performances sur benchmarks
#
# - paraphrase-multilingual-mpnet-base-v2: MULTILINGUE
#   * 768 dimensions
#   * ~1GB
#   * Support 50+ langues
#   * Bon pour français+anglais
#
# DEVICE AUTO-DÉTECTION:
# - CUDA: GPU NVIDIA (le plus rapide)
# - MPS: Apple Silicon M1/M2/M3 (rapide)
# - CPU: Fallback (plus lent mais fonctionne partout)
#
# BATCH PROCESSING:
# - Plus efficace que embed_text() en boucle
# - Batch size = 32 par défaut (bon compromis)
# - GPU: peut aller jusqu'à 128-256
# - CPU: garder 32 ou moins
#
# CACHE:
# - Évite recalcul pour textes déjà vus
# - Hash MD5 comme clé
# - Sauvegarde disque avec pickle
# - Appelé save_cache() à la fin de session
#
# SIMILARITÉ:
# - Cosine: Standard pour embeddings (0-1, insensible magnitude)
# - L2: Distance euclidienne (bon si vecteurs normalisés)
# - Dot product: Rapide mais sensible magnitude
#
# PERFORMANCE:
# - GPU: ~500-2000 texts/sec (selon modèle)
# - CPU: ~50-200 texts/sec
# - Cache améliore drastiquement si textes répétés
#
# MÉMOIRE:
# - Modèle en RAM: 80MB-1.3GB selon modèle
# - Cache: ~4KB par text embedder (si cache_embeddings=true)
# - Batch: garde tous embeddings en mémoire
#
# ALTERNATIVES:
# - OpenAI Embeddings (ada-002): Payant, très bon
# - Cohere Embeddings: Payant, bon
# - Instructor: Open source, instruction-based
#
# EXTENSIONS:
# - Fine-tuning sur corpus math spécifique
# - Embeddings multimodaux (texte + images)
# - Compression embeddings (PCA, quantization)
#
# DEBUGGING:
# ```python
# # Tester
# embedder = Embedder(config)
# vec = embedder.embed_text("test")
# print(f"Dimension: {vec.shape}")
# print(f"Norme: {np.linalg.norm(vec):.3f}")
# print(f"Min/Max: {vec.min():.3f} / {vec.max():.3f}")
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
