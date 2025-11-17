"""
Script de test pour vérifier les composants du système RAG.

Usage:
    python test_questions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vectorization.embedder import Embedder
from src.vectorization.vector_store import VectorStore
from src.utils.config_loader import load_config

def test_retrieval_component():
    """Test du composant de retrieval (modèle ouvert)."""
    print("=" * 80)
    print("TEST 1: MODÈLE OUVERT (Sentence-Transformers)")
    print("=" * 80)

    config = load_config()

    # Initialiser embedder
    print("\n📦 Chargement du modèle d'embeddings...")
    embedder = Embedder(config)
    print(f"✓ Modèle: {config.embeddings.model}")
    print(f"✓ Dimension: {embedder.embedding_dim}")

    # Initialiser vector store
    print("\n📦 Chargement du vector store...")
    vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)
    vector_store.load("default")
    print(f"✓ Vecteurs: {vector_store.total_vectors}")

    # Tester recherche
    test_questions = [
        "Qu'est-ce qu'une dérivée ?",
        "Définition d'un espace vectoriel",
        "Comment calculer une intégrale ?",
        "Quelle est la météo aujourd'hui ?"  # Hors sujet
    ]

    print("\n🔍 Test de recherche sémantique:\n")

    for question in test_questions:
        print(f"Question: \"{question}\"")

        # Générer embedding
        query_vector = embedder.embed_text(question)

        # Rechercher
        results = vector_store.search(query_vector, top_k=3)

        if results:
            print(f"  → {len(results)} documents trouvés")
            print(f"  → Score moyen: {sum(r['score'] for r in results) / len(results):.3f}")
            print(f"  → Meilleur score: {results[0]['score']:.3f}")

            # Afficher premier résultat
            first = results[0]
            text_preview = first['metadata']['text'][:100].replace('\n', ' ')
            print(f"  → Extrait: {text_preview}...")
        else:
            print("  → Aucun document trouvé")

        print()

    print("=" * 80)
    print("✅ Modèle ouvert fonctionne (retrieval local gratuit)")
    print("=" * 80)


def test_generation_component():
    """Test du composant de génération (modèle fermé)."""
    print("\n\n")
    print("=" * 80)
    print("TEST 2: MODÈLE FERMÉ (GPT-4o)")
    print("=" * 80)

    config = load_config()

    print(f"\n📦 Configuration LLM:")
    print(f"  → Provider: {config.llm.provider}")
    print(f"  → Modèle: {config.llm.model}")
    print(f"  → Température: {config.llm.temperature}")

    print("\n⚠️  Pour tester la génération:")
    print("  1. Ouvrir http://localhost:8501")
    print("  2. Poser: 'Qu'est-ce qu'une dérivée ?'")
    print("  3. Observer dans '⚙️ Détails':")
    print("     - Tokens utilisés (GPT-4o a travaillé)")
    print("     - Coût (API payante)")
    print("     - Temps de génération")

    print("\n💡 Dans les logs, vous verrez:")
    print("  - [embedder] Génération embedding (modèle ouvert)")
    print("  - [retriever] Recherche documents (modèle ouvert)")
    print("  - [generator] Appel API GPT-4o (modèle fermé)")
    print("  - [cost_tracker] Coût calculé (modèle fermé)")

    print("\n=" * 80)
    print("ℹ️  Les deux modèles sont TOUJOURS utilisés ensemble:")
    print("   - Ouvert: Embeddings + Retrieval (local, gratuit)")
    print("   - Fermé: Génération (API, payant)")
    print("=" * 80)


if __name__ == "__main__":
    print("\n🧪 TEST DES COMPOSANTS DU SYSTÈME RAG HYBRIDE\n")

    try:
        test_retrieval_component()
        test_generation_component()

        print("\n\n🎯 PROCHAINES ÉTAPES:")
        print("1. Lancer Streamlit: http://localhost:8501")
        print("2. Poser questions de test (voir ci-dessus)")
        print("3. Monitorer logs: ./monitor_logs.sh")
        print("4. Observer métriques dans sidebar\n")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
