"""
Base vectorielle FAISS pour recherche sémantique rapide.

Ce module gère le stockage et la recherche de vecteurs avec FAISS.
FAISS = Facebook AI Similarity Search (open source, très rapide).

Usage:
    from src.vectorization.vector_store import VectorStore

    store = VectorStore(config)
    store.add_vectors(embeddings, metadata_list)
    results = store.search(query_vector, top_k=5)
"""

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pickle
import json

from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import VectorStoreError

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VectorStore
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Store vectoriel basé sur FAISS.

    FAISS (Facebook AI Similarity Search):
    - Très rapide (C++, optimisé)
    - Support millions de vecteurs
    - Plusieurs types d'index (exact, approximate)
    - Open source

    ALTERNATIVES:
    - ChromaDB: Plus simple, metadata-friendly
    - Pinecone: Cloud, scalable (payant)
    - Weaviate: Production-ready, open source
    - Qdrant: Moderne, performant

    Example:
        >>> config = load_config()
        >>> store = VectorStore(config)
        >>> store.add_vectors(embeddings, [{"text": "...", "source": "..."}])
        >>> results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        config: Optional[object] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Args:
            config: Objet Config avec vector_store settings
            embedding_dim: Dimension des embeddings (si pas de config)
        """
        # Configuration
        if config and hasattr(config, 'vector_store'):
            self.index_type = config.vector_store.index_type
            self.persist_path = Path(config.vector_store.persist_path)
            self.similarity_metric = config.vector_store.similarity_metric
        else:
            self.index_type = "IndexFlatL2"
            self.persist_path = Path("data/vector_store")
            self.similarity_metric = "cosine"

        # Dimension (sera définie au premier add ou load)
        self.embedding_dim = embedding_dim

        # Index FAISS (créé plus tard)
        self.index: Optional[faiss.Index] = None

        # Métadonnées (stockées séparément car FAISS ne gère que les vecteurs)
        self.metadata_list: List[Dict] = []

        # Stats
        self.total_vectors = 0

        logger.info(
            "VectorStore initialized",
            extra={
                "index_type": self.index_type,
                "similarity_metric": self.similarity_metric,
                "persist_path": str(self.persist_path)
            }
        )

        # Créer répertoire
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Crée un index FAISS.

        Args:
            dimension: Dimension des vecteurs

        Returns:
            Index FAISS

        INDEX TYPES:
        - IndexFlatL2: Recherche exacte avec distance L2
          * Précis mais lent pour >1M vecteurs
          * Recommandé: <100k vecteurs
        - IndexFlatIP: Recherche exacte avec produit scalaire (inner product)
          * Pour cosine similarity (avec vecteurs normalisés)
        - IndexIVFFlat: Recherche approximative (Inverted File)
          * Plus rapide, moins précis
          * Recommandé: >100k vecteurs
        - IndexHNSWFlat: Graph-based (Hierarchical Navigable Small World)
          * Très rapide, bonne précision
          * Plus de RAM
        """
        if self.index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created IndexFlatL2 (exact search, L2 distance)")

        elif self.index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created IndexFlatIP (exact search, inner product)")

        elif self.index_type == "IndexIVFFlat":
            # Index avec quantization (approximate)
            # nlist = nombre de clusters (√N généralement)
            nlist = 100  # Pour commencer
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(f"Created IndexIVFFlat (approximate search, nlist={nlist})")
            logger.warning("IndexIVFFlat needs training before use!")

        elif self.index_type == "IndexHNSWFlat":
            # Graph-based index
            M = 32  # Nombre de connexions par point
            index = faiss.IndexHNSWFlat(dimension, M)
            logger.info(f"Created IndexHNSWFlat (graph-based, M={M})")

        else:
            # Fallback
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
            index = faiss.IndexFlatL2(dimension)

        return index

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadata: List[Dict],
        normalize: bool = None
    ) -> None:
        """
        Ajoute des textes avec leurs embeddings à l'index.

        Méthode de commodité qui combine textes et métadonnées.

        Args:
            texts: Liste de textes
            embeddings: Liste d'embeddings (un par texte)
            metadata: Liste de métadonnées (une par texte)
            normalize: Normaliser les vecteurs (auto si metric=cosine)

        Raises:
            VectorStoreError: Si erreur d'ajout

        Example:
            >>> texts = ["text1", "text2"]
            >>> embeddings = embedder.embed_texts(texts)
            >>> metadata = [{"source": "doc1"}, {"source": "doc2"}]
            >>> store.add_texts(texts, embeddings, metadata)
        """
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise VectorStoreError(
                f"Length mismatch: {len(texts)} texts, {len(embeddings)} embeddings, {len(metadata)} metadata"
            )

        # Ajouter texte dans métadonnées
        enriched_metadata = []
        for text, meta in zip(texts, metadata):
            meta_copy = meta.copy()
            meta_copy["text"] = text
            enriched_metadata.append(meta_copy)

        # Convertir embeddings en numpy array
        vectors = np.array(embeddings)

        # Appeler add_vectors
        self.add_vectors(vectors, enriched_metadata, normalize=normalize)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict],
        normalize: bool = None
    ) -> None:
        """
        Ajoute des vecteurs à l'index.

        Args:
            vectors: Matrice (n_vectors, dimension)
            metadata: Liste de métadonnées (une par vecteur)
            normalize: Normaliser les vecteurs (auto si metric=cosine)

        Raises:
            VectorStoreError: Si erreur d'ajout

        Example:
            >>> embeddings = embedder.embed_texts(["text1", "text2"])
            >>> metadata = [{"text": "text1"}, {"text": "text2"}]
            >>> store.add_vectors(embeddings, metadata)
        """
        if len(vectors) != len(metadata):
            raise VectorStoreError(
                f"Mismatch: {len(vectors)} vectors but {len(metadata)} metadata"
            )

        # Créer index si pas encore fait
        if self.index is None:
            dimension = vectors.shape[1]
            self.embedding_dim = dimension
            self.index = self._create_index(dimension)

        # Vérifier dimension
        if vectors.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Dimension mismatch: expected {self.embedding_dim}, got {vectors.shape[1]}"
            )

        # Normaliser si cosine similarity
        if normalize is None:
            normalize = (self.similarity_metric == "cosine")

        if normalize:
            vectors = self._normalize_vectors(vectors)
            logger.debug("✓ Vectors normalized for cosine similarity")

        # Ajouter à FAISS
        try:
            with log_performance(logger, f"add_{len(vectors)}_vectors_to_index"):
                # Convertir en float32 (requis par FAISS)
                vectors_f32 = vectors.astype(np.float32)

                self.index.add(vectors_f32)

                # Ajouter métadonnées
                self.metadata_list.extend(metadata)

                self.total_vectors += len(vectors)

                logger.info(
                    f"✓ Added {len(vectors)} vectors to index",
                    extra={
                        "added": len(vectors),
                        "total": self.total_vectors
                    }
                )

        except Exception as e:
            raise VectorStoreError(
                f"Failed to add vectors to index: {e}",
                details={"n_vectors": len(vectors)}
            ) from e

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Recherche les vecteurs les plus similaires.

        Args:
            query_vector: Vecteur de requête (dimension,)
            top_k: Nombre de résultats
            similarity_threshold: Seuil de similarité minimum (optionnel)
            filters: Filtres sur métadonnées (optionnel, basique)

        Returns:
            Liste de résultats triés par similarité

        Format de retour:
        [
            {
                "index": 42,
                "score": 0.89,
                "distance": 0.45,  # Distance brute FAISS
                "metadata": {...}
            },
            ...
        ]

        Example:
            >>> query_vec = embedder.embed_text("Qu'est-ce qu'une dérivée ?")
            >>> results = store.search(query_vec, top_k=5)
            >>> for res in results:
            ...     print(f"Score: {res['score']:.3f} - {res['metadata']['text'][:50]}")
        """
        if self.index is None or self.total_vectors == 0:
            logger.warning("Empty index, no search results")
            return []

        # Vérifier dimension
        if query_vector.shape[0] != self.embedding_dim:
            raise VectorStoreError(
                f"Query dimension mismatch: expected {self.embedding_dim}, got {query_vector.shape[0]}"
            )

        # Normaliser si cosine
        if self.similarity_metric == "cosine":
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1))[0]

        try:
            with log_performance(logger, "vector_search"):
                # Convertir en float32
                query_f32 = query_vector.astype(np.float32).reshape(1, -1)

                # Recherche FAISS
                # distances: plus petit = plus similaire (L2)
                # Pour inner product (IndexFlatIP): plus grand = plus similaire
                distances, indices = self.index.search(query_f32, top_k)

                # Convertir en liste de résultats
                results = []

                for i in range(len(indices[0])):
                    idx = int(indices[0][i])
                    distance = float(distances[0][i])

                    # Ignorer résultats invalides
                    if idx < 0 or idx >= len(self.metadata_list):
                        continue

                    # Convertir distance en score de similarité (0-1)
                    score = self._distance_to_similarity(distance)

                    # Filtrer par seuil si spécifié
                    if similarity_threshold and score < similarity_threshold:
                        continue

                    result = {
                        "index": idx,
                        "score": score,
                        "distance": distance,
                        "metadata": self.metadata_list[idx]
                    }

                    # Appliquer filtres basiques sur métadonnées
                    if filters:
                        if not self._matches_filters(result["metadata"], filters):
                            continue

                    results.append(result)

                logger.debug(
                    f"✓ Search returned {len(results)} results",
                    extra={"top_k": top_k, "results": len(results)}
                )

                return results

        except Exception as e:
            raise VectorStoreError(
                f"Search failed: {e}",
                details={"top_k": top_k}
            ) from e

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convertit une distance FAISS en score de similarité (0-1).

        Args:
            distance: Distance brute de FAISS

        Returns:
            Score de similarité (1 = identique, 0 = très différent)
        """
        if self.index_type == "IndexFlatIP":
            # Inner product: déjà un score (plus grand = meilleur)
            # Normaliser entre 0 et 1
            return min(max(distance, 0.0), 1.0)

        else:
            # L2 distance: plus petit = meilleur
            # Convertir en similarité
            # Formule: 1 / (1 + distance)
            return 1 / (1 + distance)

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """
        Vérifie si métadonnées matchent les filtres.

        Filtres basiques (égalité stricte).

        Args:
            metadata: Métadonnées d'un résultat
            filters: Filtres requis

        Returns:
            True si match

        Example:
            >>> filters = {"level": "L2", "source": "cours.pdf"}
            >>> metadata = {"level": "L2", "source": "cours.pdf", "page": 5}
            >>> _matches_filters(metadata, filters)  # True
        """
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalise les vecteurs (norme L2 = 1).

        Pour cosine similarity avec IndexFlatIP.

        Args:
            vectors: Matrice (n, dim)

        Returns:
            Vecteurs normalisés
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Éviter division par zéro
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def save(self, name: str = "default") -> None:
        """
        Sauvegarde l'index et métadonnées sur disque.

        Args:
            name: Nom de l'index (pour plusieurs index)

        Example:
            >>> store.save("math_docs")
        """
        if self.index is None:
            logger.warning("No index to save (empty)")
            return

        index_path = self.persist_path / f"{name}.index"
        metadata_path = self.persist_path / f"{name}.metadata"

        try:
            # Sauvegarder index FAISS
            faiss.write_index(self.index, str(index_path))

            # Sauvegarder métadonnées
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    "metadata_list": self.metadata_list,
                    "embedding_dim": self.embedding_dim,
                    "total_vectors": self.total_vectors,
                    "similarity_metric": self.similarity_metric,
                    "index_type": self.index_type
                }, f)

            logger.info(
                f"✓ Saved index to {index_path}",
                extra={
                    "index_path": str(index_path),
                    "vectors": self.total_vectors
                }
            )

        except Exception as e:
            raise VectorStoreError(
                f"Failed to save index: {e}",
                details={"path": str(index_path)}
            ) from e

    def load(self, name: str = "default") -> None:
        """
        Charge l'index et métadonnées depuis le disque.

        Args:
            name: Nom de l'index

        Raises:
            VectorStoreError: Si fichiers introuvables

        Example:
            >>> store.load("math_docs")
        """
        index_path = self.persist_path / f"{name}.index"
        metadata_path = self.persist_path / f"{name}.metadata"

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(
                f"Index files not found: {index_path}",
                details={"index_path": str(index_path)}
            )

        try:
            # Charger index FAISS
            self.index = faiss.read_index(str(index_path))

            # Charger métadonnées
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)

            self.metadata_list = data["metadata_list"]
            self.embedding_dim = data["embedding_dim"]
            self.total_vectors = data["total_vectors"]
            # Pas override similarity_metric et index_type (déjà dans config)

            logger.info(
                f"✓ Loaded index from {index_path}",
                extra={
                    "vectors": self.total_vectors,
                    "dimension": self.embedding_dim
                }
            )

        except Exception as e:
            raise VectorStoreError(
                f"Failed to load index: {e}",
                details={"path": str(index_path)}
            ) from e

    def get_stats(self) -> Dict:
        """
        Retourne des statistiques sur l'index.

        Returns:
            Dict avec statistiques
        """
        return {
            "total_vectors": self.total_vectors,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "similarity_metric": self.similarity_metric,
            "is_trained": self.index.is_trained if self.index else False,
            "persist_path": str(self.persist_path)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# FAISS INDEX TYPES:
#
# IndexFlatL2 (DÉFAUT):
#   ✓ Recherche exacte
#   ✓ Distance L2 (euclidienne)
#   ✓ Simple et fiable
#   ✗ Lent pour >100k vecteurs
#   Use: <100k vecteurs, précision importante
#
# IndexFlatIP:
#   ✓ Recherche exacte
#   ✓ Inner product (produit scalaire)
#   ✓ Pour cosine similarity (avec vecteurs normalisés)
#   Use: Cosine similarity requis
#
# IndexIVFFlat:
#   ✓ Recherche approximative (plus rapide)
#   ✓ Bonne précision (90-95%)
#   ✗ Nécessite training (build avec sample de vecteurs)
#   Use: >100k vecteurs, vitesse importante
#
# IndexHNSWFlat:
#   ✓ Très rapide
#   ✓ Bonne précision (95%+)
#   ✗ Plus de RAM
#   Use: Production, millions de vecteurs
#
# SIMILARITÉ:
# - cosine: Insensible à la magnitude (recommandé pour texte)
#   * Normaliser vecteurs
#   * Utiliser IndexFlatIP
# - l2: Distance euclidienne (sensible magnitude)
#   * Utiliser IndexFlatL2
#
# MÉTADONNÉES:
# - FAISS ne stocke QUE les vecteurs (nombres)
# - Métadonnées stockées séparément (liste Python)
# - Index FAISS ↔ Index métadonnées (synchronisés)
# - Filtres métadonnées: basiques (égalité) pour MVP
#   * Pour filtres complexes: utiliser ChromaDB ou Weaviate
#
# PERSISTANCE:
# - Index: Fichier binaire FAISS (.index)
# - Métadonnées: Pickle (.metadata)
# - Save régulièrement (après add batch)
#
# PERFORMANCE:
# - IndexFlatL2: ~1000 queries/sec (100k vecteurs, CPU)
# - IndexIVFFlat: ~5000 queries/sec (1M vecteurs, après training)
# - IndexHNSWFlat: ~10000 queries/sec (1M vecteurs)
# - GPU: 10-100x plus rapide (si faiss-gpu)
#
# ALTERNATIVES À FAISS:
#
# ChromaDB:
#   ✓ Métadonnées riches
#   ✓ Filtres complexes
#   ✓ API simple
#   ✗ Plus lent que FAISS
#   Use: Métadonnées importantes, filtres complexes
#
# Qdrant:
#   ✓ Production-ready
#   ✓ Filtres, clustering
#   ✓ API REST
#   ✗ Nécessite serveur
#   Use: Déploiement production
#
# Pinecone:
#   ✓ Managed cloud
#   ✓ Scalable
#   ✗ Payant
#   Use: Production cloud, pas de gestion infra
#
# DEBUGGING:
# ```python
# # Vérifier index
# store = VectorStore(config)
# print(store.get_stats())
#
# # Tester recherche
# query_vec = np.random.rand(384).astype(np.float32)
# results = store.search(query_vec, top_k=5)
# print(f"Found {len(results)} results")
# ```
#
# EXTENSIONS:
# - Support GPU (faiss-gpu)
# - Training pour IndexIVFFlat
# - Filtres métadonnées avancés (SQL-like)
# - Compression vecteurs (PQ, SQ)
# - Hybrid search (vecteurs + keywords)
#
# ═══════════════════════════════════════════════════════════════════════════════
