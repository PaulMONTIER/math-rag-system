"""
Agent Retriever - Recherche de documents pertinents (RAG).

Cet agent récupère les documents les plus pertinents pour une question
en utilisant la recherche vectorielle FAISS.

Usage:
    from src.agents.retriever import RetrieverAgent

    retriever = RetrieverAgent(config, vector_store, embedder)
    results = retriever.retrieve("Qu'est-ce qu'une dérivée ?")
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from src.vectorization.vector_store import VectorStore
from src.vectorization.embedder import Embedder
from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import AgentError

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """
    Résultat de recherche avec document et score.

    Attributes:
        text: Texte du chunk récupéré
        score: Score de similarité (0-1)
        metadata: Métadonnées (source, page, etc.)
        rank: Rang dans les résultats (1=meilleur)
    """
    text: str
    score: float
    metadata: Dict
    rank: int


class RetrieverAgent:
    """
    Agent de récupération de documents (RAG).

    Processus:
    1. Embedder la question
    2. Rechercher dans vector store (FAISS)
    3. Optionnel: Re-ranking avec cross-encoder
    4. Filtrer par seuil de similarité
    5. Retourner top_k résultats

    Example:
        >>> retriever = RetrieverAgent(config, vector_store, embedder)
        >>> results = retriever.retrieve("dérivée de x²")
        >>> for res in results:
        ...     print(f"{res.score:.2f}: {res.text[:50]}...")
    """

    def __init__(
        self,
        config: object,
        vector_store: VectorStore,
        embedder: Embedder
    ):
        """
        Args:
            config: Objet Config
            vector_store: VectorStore avec documents indexés
            embedder: Embedder pour vectoriser la question
        """
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder

        # Configuration retrieval
        if hasattr(config, 'retrieval'):
            self.top_k = config.retrieval.top_k
            self.similarity_threshold = config.retrieval.similarity_threshold
            self.use_reranking = config.retrieval.use_reranking
            self.max_context_tokens = config.retrieval.max_context_tokens
        else:
            self.top_k = 5
            self.similarity_threshold = 0.7
            self.use_reranking = False
            self.max_context_tokens = 3000

        logger.info(
            "RetrieverAgent initialized",
            extra={
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "use_reranking": self.use_reranking
            }
        )

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Récupère les documents pertinents pour une question.

        Args:
            question: Question de l'utilisateur
            top_k: Override top_k (optionnel)
            filters: Filtres métadonnées (optionnel)

        Returns:
            Liste de RetrievalResult triés par pertinence

        Example:
            >>> results = retriever.retrieve("Qu'est-ce qu'une dérivée ?")
            >>> print(f"{len(results)} documents trouvés")
            >>> print(f"Meilleur score: {results[0].score:.3f}")
        """
        if not question or not question.strip():
            raise AgentError("Question vide fournie au retriever", agent_name="retriever")

        k = top_k or self.top_k

        logger.info(
            f"Retrieving documents for question",
            extra={"question_length": len(question), "top_k": k}
        )

        try:
            # 1. Embedder la question
            with log_performance(logger, "embed_question"):
                question_vector = self.embedder.embed_text(question)

            # 2. Recherche vectorielle
            with log_performance(logger, "vector_search"):
                raw_results = self.vector_store.search(
                    query_vector=question_vector,
                    top_k=k,
                    similarity_threshold=self.similarity_threshold,
                    filters=filters
                )

            if not raw_results:
                logger.warning("No documents found above similarity threshold")
                return []

            # 3. Re-ranking (optionnel)
            if self.use_reranking:
                raw_results = self._rerank_results(question, raw_results)

            # 4. Convertir en RetrievalResult
            results = []
            for i, raw_res in enumerate(raw_results[:k], start=1):
                result = RetrievalResult(
                    text=raw_res["metadata"].get("text", ""),
                    score=raw_res["score"],
                    metadata=raw_res["metadata"],
                    rank=i
                )
                results.append(result)

            logger.info(
                f"✓ Retrieved {len(results)} documents",
                extra={
                    "count": len(results),
                    "avg_score": sum(r.score for r in results) / len(results) if results else 0
                }
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise AgentError(
                f"Document retrieval failed: {e}",
                agent_name="retriever"
            ) from e

    def calculate_retrieval_confidence(
        self,
        results: List[RetrievalResult],
        good_score_threshold: float = 0.7  # Baissé de 0.8 à 0.7
    ) -> float:
        """
        Calcule un score de confiance pour les résultats de retrieval.

        Score basé sur:
        - Score moyen (weight: 0.60) - Composante principale
        - Nombre de résultats (weight: 0.15)
        - Proportion de résultats "bons" (weight: 0.25)
        - Bonus si meilleur résultat excellent (>= 0.85)

        Args:
            results: Résultats de retrieval
            good_score_threshold: Seuil pour considérer un résultat comme "bon"

        Returns:
            float: Score de confiance entre 0.0 et 1.0
        """
        if not results:
            return 0.0

        # 1. Score moyen normalisé (0-1)
        avg_score = sum(r.score for r in results) / len(results)
        avg_component = min(avg_score, 1.0)

        # 2. Composante nombre de résultats (0-1)
        # On considère que 3+ résultats = confiance maximale pour cette composante
        count_component = min(len(results) / 3.0, 1.0)

        # 3. Proportion de résultats au-dessus du seuil "bon"
        good_results = sum(1 for r in results if r.score >= good_score_threshold)
        quality_component = good_results / len(results)

        # 4. Bonus si le meilleur résultat est excellent (>= 0.85)
        best_score = max(r.score for r in results)
        best_boost = 0.1 if best_score >= 0.85 else 0.0

        # Score final pondéré (ajusté pour favoriser le score moyen)
        base_confidence = (
            0.60 * avg_component +  # Plus de poids sur le score moyen
            0.15 * count_component +  # Moins de poids sur la quantité
            0.25 * quality_component  # Poids modéré sur la qualité
        )

        # Appliquer le bonus (plafonné à 1.0)
        confidence = min(base_confidence + best_boost, 1.0)

        logger.debug(
            f"Retrieval confidence calculated",
            extra={
                "avg_score": avg_score,
                "best_score": best_score,
                "count": len(results),
                "good_results": good_results,
                "base_confidence": base_confidence,
                "confidence": confidence
            }
        )

        return confidence

    def build_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        Construit le contexte formaté pour le LLM depuis les résultats.

        Args:
            results: Résultats de retrieve()
            include_metadata: Inclure source/page dans contexte

        Returns:
            Contexte formaté prêt pour le LLM

        Format:
        ```
        Voici les informations pertinentes trouvées :

        [Document 1] (Score: 0.89)
        Source: Analyse_L2.pdf, Page: 45
        La dérivée d'une fonction mesure...

        [Document 2] (Score: 0.85)
        Source: Calcul_L3.pdf, Page: 12
        ...
        ```

        Example:
            >>> context = retriever.build_context(results)
            >>> print(len(context), "caractères")
        """
        if not results:
            return "Aucune information pertinente trouvée dans la base de connaissances."

        context_parts = ["Voici les informations pertinentes trouvées :\n"]

        for result in results:
            # En-tête document
            header = f"\n[Document {result.rank}] (Score: {result.score:.2f})"

            if include_metadata:
                # Ajouter source et page si disponible
                source = result.metadata.get("source", "Unknown")
                page = result.metadata.get("page", "N/A")
                header += f"\nSource: {source}, Page: {page}"

            context_parts.append(header)
            context_parts.append(result.text)
            context_parts.append("")  # Ligne vide

        full_context = "\n".join(context_parts)

        # Vérifier limite tokens (approximation)
        from src.utils.cost_tracker import estimate_tokens
        context_tokens = estimate_tokens(full_context)

        if context_tokens > self.max_context_tokens:
            logger.warning(
                f"Context too long ({context_tokens} tokens), truncating to {self.max_context_tokens}",
                extra={"context_tokens": context_tokens, "max": self.max_context_tokens}
            )

            # Tronquer en gardant les meilleurs documents
            truncated_results = []
            current_tokens = 0

            for result in results:
                result_tokens = estimate_tokens(result.text)

                if current_tokens + result_tokens > self.max_context_tokens:
                    break

                truncated_results.append(result)
                current_tokens += result_tokens

            full_context = self.build_context(truncated_results, include_metadata)

        return full_context

    def _rerank_results(
        self,
        question: str,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Re-ranking avec cross-encoder (optionnel).

        CROSS-ENCODER:
        - Plus précis que bi-encoder (embeddings)
        - Calcule score (question, document) directement
        - Plus lent (doit traiter chaque paire)

        Args:
            question: Question
            results: Résultats bruts de FAISS

        Returns:
            Résultats re-rankés

        Note:
            Nécessite cross-encoder installé et configuré.
            Pas implémenté par défaut (TODO).
        """
        logger.warning("Re-ranking not implemented yet, returning original order")

        # TODO: Implémenter avec cross-encoder
        # from sentence_transformers import CrossEncoder
        # reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        #
        # pairs = [(question, res["metadata"]["text"]) for res in results]
        # scores = reranker.predict(pairs)
        #
        # # Re-trier par scores
        # reranked = sorted(
        #     zip(results, scores),
        #     key=lambda x: x[1],
        #     reverse=True
        # )
        # return [res for res, score in reranked]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiques du retriever.

        Returns:
            Dict avec stats
        """
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "use_reranking": self.use_reranking,
            "max_context_tokens": self.max_context_tokens,
            "vector_store_stats": self.vector_store.get_stats()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# RAG (Retrieval-Augmented Generation):
# 1. Retrieve: Chercher documents pertinents (ce module)
# 2. Augment: Construire contexte pour LLM
# 3. Generate: LLM génère réponse avec contexte (generator.py)
#
# PROCESSUS RETRIEVAL:
# 1. Question → Embedder → Vecteur
# 2. Vecteur → FAISS search → Top K résultats
# 3. Filtrer par similarity_threshold
# 4. (Optionnel) Re-ranking avec cross-encoder
# 5. Retourner résultats triés
#
# TOP_K:
# - 5 par défaut (bon compromis)
# - Augmenter (7-10) si questions complexes
# - Diminuer (3) si veux réponses très focused
# - Trade-off: Plus de K = plus de contexte mais plus de bruit
#
# SIMILARITY_THRESHOLD:
# - 0.7 par défaut (filtre résultats peu pertinents)
# - Augmenter (0.8) pour plus de précision
# - Diminuer (0.6) si pas assez de résultats
# - 1.0 = identique, 0.0 = pas du tout similaire
#
# RE-RANKING:
# - Cross-encoder plus précis que bi-encoder
# - Bi-encoder: Question→Vec, Doc→Vec, compare vecteurs (rapide)
# - Cross-encoder: (Question, Doc)→Score (précis mais lent)
# - Utiliser si précision critique et top_k petit (<10)
#
# CONTEXTE POUR LLM:
# - Format structuré avec sources
# - Limite max_context_tokens (éviter dépassement LLM)
# - Tronquer si nécessaire (garder meilleurs docs)
# - Citations facilitent traçabilité
#
# MÉTADONNÉES:
# - source: Nom du PDF
# - page: Numéro de page
# - section: Chapitre/section (si détecté)
# - level: Niveau étudiant (L1/L2/L3/M1/M2)
# - has_formula: Si contient formules LaTeX
#
# FILTRES:
# - Filtrer par métadonnées (ex: level="L2", source="cours.pdf")
# - Implémentation basique dans VectorStore
# - Pour filtres avancés: utiliser ChromaDB ou Qdrant
#
# PERFORMANCE:
# - Embedding question: ~10-50ms (CPU)
# - FAISS search: ~5-20ms (100k vecteurs, CPU)
# - Re-ranking: ~100-500ms (cross-encoder, 10 docs)
# - Total: ~100-600ms selon configuration
#
# EXTENSIONS:
# - Hybrid search (vecteurs + keywords BM25)
# - Query expansion (reformuler question)
# - Multi-query (générer plusieurs reformulations)
# - Filtrage sémantique (éliminer contradictions)
#
# DEBUGGING:
# ```python
# # Tester
# retriever = RetrieverAgent(config, vector_store, embedder)
# results = retriever.retrieve("dérivée")
# for res in results:
#     print(f"Score: {res.score:.3f}")
#     print(f"Text: {res.text[:100]}...")
#     print(f"Source: {res.metadata.get('source')}")
#     print()
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
