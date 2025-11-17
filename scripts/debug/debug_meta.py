"""Debug: Afficher les métadonnées des documents récupérés."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.vectorization.embedder import Embedder
from src.vectorization.vector_store import VectorStore
from src.agents.retriever import RetrieverAgent
from src.utils.config_loader import load_config

config = load_config()

# Initialiser composants
embedder = Embedder(config)
vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)
vector_store.load("default")

# Retriever
retriever = RetrieverAgent(config, vector_store, embedder)

# Tester
question = "Qu'est-ce qu'une dérivée ?"
results = retriever.retrieve(question, top_k=5)

print(f"\n{'='*80}")
print(f"QUESTION: {question}")
print(f"{'='*80}\n")

for i, result in enumerate(results, 1):
    print(f"[Document {i}] Score: {result.score:.3f}")
    print(f"  Métadonnées complètes:")
    for key, value in result.metadata.items():
        if key == "text":
            print(f"    {key}: {str(value)[:50]}...")
        else:
            print(f"    {key}: {value}")
    print()
