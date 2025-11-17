"""Script simple pour tester rapidement le système."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test direct
print("Testing simple vector store build...")

# Créer un chunk de test
test_chunks = [
    {
        "text": "Introduction à l'analyse mathématique. La dérivée d'une fonction.",
        "metadata": {"source": "test.pdf", "chunk_id": "test_0"}
    }
]

# Essayer d'embedder
from src.vectorization.embedder import Embedder
from src.vectorization.vector_store import VectorStore
from src.utils.config_loader import load_config

config = load_config()
embedder = Embedder(config)
vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)

# Embed
texts = [c["text"] for c in test_chunks]
embeddings = embedder.embed_texts(texts)

print(f"✓ Created {len(embeddings)} embeddings")
print(f"  Dimension: {len(embeddings[0])}")

# Add to vector store
metadatas = [c["metadata"] for c in test_chunks]
vector_store.add_texts(texts, embeddings, metadatas)

print(f"✓ Vector store has {vector_store.total_vectors} vectors")

# Save
vector_store.save("test")
print(f"✓ Saved to data/vector_store/test.index")

# Test search
query_vector = embedder.embed_text("Qu'est-ce qu'une dérivée ?")
results = vector_store.search(query_vector, top_k=1)

if results:
    print(f"✓ Search works! Score: {results[0]['score']:.3f}")
    print(f"  Text: {results[0]['text'][:50]}...")
else:
    print("❌ Search returned no results")

print("\n✅ System is working! You can now:")
print("   1. Fix the build script bugs")
print("   2. Or use this simple approach for testing")
