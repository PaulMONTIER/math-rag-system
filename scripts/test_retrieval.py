"""
Script de test de retrieval.

Teste la recherche vectorielle pour vérifier que:
1. Le vector store se charge correctement
2. Les embeddings sont générés correctement
3. La recherche retourne des résultats pertinents
4. Les formules LaTeX sont préservées

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --query "Qu'est-ce qu'une dérivée ?"
"""

import sys
from pathlib import Path
import argparse

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorization.vector_store import VectorStore
from src.vectorization.embedder import Embedder
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_banner():
    """Affiche le banner."""
    print()
    print("═" * 80)
    print("  TEST DE RETRIEVAL VECTORIEL")
    print("═" * 80)
    print()


def print_result(result: dict, rank: int):
    """Affiche un résultat de recherche de façon lisible."""
    print(f"\n{'─' * 80}")
    print(f"Rank #{rank} - Score: {result['score']:.4f}")
    print(f"{'─' * 80}")

    # Métadonnées
    metadata = result.get('metadata', {})
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Chunk ID: {metadata.get('chunk_id', 'Unknown')}")

    if 'num_formulas' in metadata:
        print(f"Formulas: {metadata['num_formulas']}")

    if 'char_count' in metadata:
        print(f"Length: {metadata['char_count']} chars")

    # Texte (tronqué si trop long)
    text = result.get('text', '')
    max_len = 500

    print(f"\nText:")
    if len(text) > max_len:
        print(f"{text[:max_len]}...")
        print(f"[... {len(text) - max_len} more characters]")
    else:
        print(text)


def test_queries(
    embedder: Embedder,
    vector_store: VectorStore,
    queries: list,
    top_k: int = 3
):
    """Teste plusieurs requêtes."""
    for i, query in enumerate(queries, 1):
        print("\n" + "═" * 80)
        print(f"Query {i}/{len(queries)}: {query}")
        print("═" * 80)

        # Générer embedding de la question
        query_vector = embedder.embed_text(query)

        # Rechercher
        results = vector_store.search(
            query_vector,
            top_k=top_k,
            similarity_threshold=0.0  # Pas de filtrage pour test
        )

        if not results:
            print("⚠️  No results found")
            continue

        print(f"\nFound {len(results)} results:")

        # Afficher résultats
        for rank, result in enumerate(results, 1):
            print_result(result, rank)

        print()


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Test vector retrieval"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to test (optional)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to retrieve (default: 3)"
    )

    args = parser.parse_args()

    print_banner()

    try:
        # 1. Charger configuration
        print("Loading configuration...")
        config = load_config()
        print("✓ Configuration loaded")
        print()

        # 2. Initialiser embedder
        print("Initializing embedder...")
        embedder = Embedder(config)
        print(f"✓ Embedder ready (model: {config.embeddings.model})")
        print(f"  Embedding dimension: {embedder.embedding_dim}")
        print(f"  Device: {embedder.device}")
        print()

        # 3. Charger vector store
        print("Loading vector store...")
        vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)

        try:
            vector_store.load("default")
            print(f"✓ Vector store loaded")
            print(f"  Total vectors: {vector_store.total_vectors}")
            print()
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            print()
            print("Have you built the vector store yet?")
            print("  Run: python scripts/build_vector_store.py")
            return

        if vector_store.total_vectors == 0:
            print("⚠️  Vector store is empty!")
            print("  Run: python scripts/build_vector_store.py")
            return

        # 4. Préparer requêtes de test
        if args.query:
            queries = [args.query]
        else:
            # Requêtes par défaut
            queries = [
                "Qu'est-ce qu'une dérivée ?",
                "Comment calculer une intégrale ?",
                "Qu'est-ce qu'un espace vectoriel ?",
                "Définition de la continuité",
                "Théorème de Pythagore"
            ]

        print(f"Testing {len(queries)} queries with top_k={args.top_k}")
        print()

        # 5. Tester
        test_queries(embedder, vector_store, queries, top_k=args.top_k)

        # 6. Résumé
        print("═" * 80)
        print("  TEST COMPLETE!")
        print("═" * 80)
        print()
        print("If results look good, you're ready to:")
        print("  1. Launch interface: make run")
        print("  2. Or: streamlit run src/interface/app.py")
        print()

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        logger.info("Test cancelled by user")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
        print("   Check logs for details: data/logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# OBJECTIF:
# Tester que le retrieval vectoriel fonctionne correctement.
# Valider que les formules LaTeX sont préservées dans les résultats.
#
# TESTS:
# 1. Charger vector store
# 2. Générer embeddings de questions test
# 3. Rechercher dans vector store
# 4. Afficher résultats avec scores
# 5. Vérifier que formulas LaTeX apparaissent intactes
#
# QUERIES PAR DÉFAUT:
# - Questions mathématiques de base
# - Couvrent différents sujets (analyse, algèbre, géométrie)
# - Permettent de vérifier pertinence
#
# OUTPUT:
# - Score de similarité
# - Métadonnées (source, chunk_id, num_formulas)
# - Texte (tronqué si trop long)
# - Check visuel des formulas LaTeX
#
# USAGE:
# ```bash
# # Test avec queries par défaut
# python scripts/test_retrieval.py
#
# # Test avec query custom
# python scripts/test_retrieval.py --query "Qu'est-ce qu'une limite ?"
#
# # Plus de résultats
# python scripts/test_retrieval.py --top-k 5
# ```
#
# VALIDATION:
# - Scores doivent être entre 0 et 1
# - Résultats pertinents doivent avoir score > 0.5
# - Formules LaTeX doivent apparaître avec $ ou \[ délimiteurs
# - Sources doivent être identifiées
#
# DEBUGGING:
# - Si pas de résultats: vector store vide ou mal construit
# - Si scores très bas: embeddings incompatibles ou questions hors sujet
# - Si formulas coupées: BUG CRITIQUE dans chunker → reporter
#
# ═══════════════════════════════════════════════════════════════════════════════
