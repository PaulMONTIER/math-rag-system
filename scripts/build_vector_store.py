"""
Script de construction de la base vectorielle.

Ce script:
1. Scan le dossier data/raw/ pour tous les PDFs
2. Extrait le texte de chaque PDF
3. Découpe en chunks (SANS couper les formules LaTeX)
4. Génère les embeddings
5. Stocke dans FAISS
6. Sauvegarde l'index

Usage:
    python scripts/build_vector_store.py
    python scripts/build_vector_store.py --rebuild  # Force reconstruction

Output:
    data/vector_store/default.index (FAISS index)
    data/vector_store/default.metadata (métadonnées)
    data/logs/build_vector_store.log (logs)
"""

import sys
from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_processor import PDFProcessor
from src.vectorization.chunker import Chunker
from src.vectorization.embedder import Embedder
from src.vectorization.vector_store import VectorStore
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.exceptions import PDFExtractionError, ChunkingError, EmbeddingError

logger = get_logger(__name__)


def find_pdf_files(directory: Path) -> List[Path]:
    """
    Trouve tous les fichiers PDF dans un dossier.

    Args:
        directory: Dossier à scanner

    Returns:
        Liste de chemins vers les PDFs
    """
    pdf_files = list(directory.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files


def process_pdfs(
    pdf_files: List[Path],
    pdf_processor: PDFProcessor,
    chunker: Chunker
) -> List[dict]:
    """
    Traite tous les PDFs et génère les chunks.

    Args:
        pdf_files: Liste de PDFs à traiter
        pdf_processor: Processeur PDF
        chunker: Chunker de texte

    Returns:
        Liste de chunks avec métadonnées
    """
    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            logger.info(f"Processing: {pdf_path.name}")

            # 1. Extraire contenu du PDF
            pdf_doc = pdf_processor.process_pdf(pdf_path)

            logger.info(
                f"  Extracted {len(pdf_doc.pages)} pages, "
                f"{len(pdf_doc.formulas)} formulas"
            )

            # 2. Chunker PAGE PAR PAGE pour tracker les numéros de page
            page_chunks = []
            for page_num, page_text in pdf_doc.pages.items():
                chunks_for_page = chunker.chunk_text(
                    text=page_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,  # ← AJOUT DU NUMÉRO DE PAGE!
                        "num_pages": len(pdf_doc.pages),
                        "has_formulas": len(pdf_doc.formulas) > 0
                    }
                )
                page_chunks.extend(chunks_for_page)

            logger.info(f"  Generated {len(page_chunks)} chunks from {len(pdf_doc.pages)} pages")

            # 3. Ajouter métadonnées supplémentaires
            for chunk in page_chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "num_formulas": chunk.num_formulas,
                        "char_count": chunk.char_count
                    }
                }
                all_chunks.append(chunk_dict)

            # VALIDATION CRITIQUE: Vérifier que les formules ne sont PAS coupées
            if pdf_doc.formulas:
                for formula in pdf_doc.formulas:
                    formula_found = False
                    for chunk in page_chunks:
                        if formula.content in chunk.text:
                            formula_found = True
                            break

                    if not formula_found:
                        logger.warning(
                            f"⚠️ Formula may be split: {formula.content[:50]}..."
                        )

        except PDFExtractionError as e:
            logger.error(f"Failed to extract {pdf_path.name}: {e}")
            continue

        except ChunkingError as e:
            logger.error(f"Failed to chunk {pdf_path.name}: {e}")
            continue

        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path.name}: {e}")
            continue

    logger.info(f"✓ Total chunks generated: {len(all_chunks)}")
    return all_chunks


def build_vector_store(
    chunks: List[dict],
    embedder: Embedder,
    vector_store: VectorStore,
    batch_size: int = 32
) -> None:
    """
    Construit la base vectorielle à partir des chunks.

    Args:
        chunks: Liste de chunks avec métadonnées
        embedder: Embedder pour générer vecteurs
        vector_store: Vector store pour stocker
        batch_size: Taille des batchs pour embedding
    """
    logger.info("Building vector store...")

    # 1. Extraire textes
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # 2. Générer embeddings (par batch pour performance)
    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]

        try:
            batch_embeddings = embedder.embed_texts(batch_texts, show_progress=False)
            all_embeddings.extend(batch_embeddings)

        except EmbeddingError as e:
            logger.error(f"Failed to embed batch {i // batch_size}: {e}")
            raise

    logger.info(f"✓ Generated {len(all_embeddings)} embeddings")

    # 3. Ajouter au vector store
    logger.info("Adding to vector store...")
    vector_store.add_texts(texts, all_embeddings, metadatas)

    logger.info(f"✓ Vector store built with {vector_store.total_vectors} vectors")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Build vector store from PDFs"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/raw",
        help="Directory containing PDFs (default: data/raw)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)"
    )

    args = parser.parse_args()

    # Banner
    print("═" * 80)
    print("  CONSTRUCTION DE LA BASE VECTORIELLE")
    print("═" * 80)
    print()

    try:
        # 1. Charger configuration
        logger.info("Loading configuration...")
        config = load_config()

        # 2. Initialiser composants
        logger.info("Initializing components...")
        pdf_processor = PDFProcessor(config)
        chunker = Chunker(config)
        embedder = Embedder(config)
        vector_store = VectorStore(config, embedding_dim=embedder.embedding_dim)

        # 3. Vérifier si index existe déjà
        index_path = Path(config.vector_store.persist_path) / "default.index"
        if index_path.exists() and not args.rebuild:
            print(f"⚠️  Vector store already exists: {index_path}")
            response = input("Rebuild anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return

        # 4. Trouver PDFs
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            print(f"❌ Directory not found: {pdf_dir}")
            print(f"   Please add PDF files to: {pdf_dir.absolute()}")
            return

        pdf_files = find_pdf_files(pdf_dir)

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            print(f"⚠️  No PDF files found in: {pdf_dir}")
            print(f"   Please add some PDF files first.")
            return

        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        print()

        # 5. Traiter PDFs et générer chunks
        print("Step 1/3: Processing PDFs and chunking text...")
        chunks = process_pdfs(pdf_files, pdf_processor, chunker)

        if not chunks:
            logger.error("No chunks generated!")
            print("❌ Failed to generate chunks from PDFs")
            return

        print(f"✓ Generated {len(chunks)} chunks")
        print()

        # 6. Construire vector store
        print("Step 2/3: Generating embeddings and building vector store...")
        build_vector_store(chunks, embedder, vector_store, batch_size=args.batch_size)
        print()

        # 7. Sauvegarder
        print("Step 3/3: Saving vector store...")
        vector_store.save("default")
        print(f"✓ Saved to: {index_path}")
        print()

        # 8. Statistiques finales
        print("═" * 80)
        print("  BUILD COMPLETE!")
        print("═" * 80)
        print(f"Total PDFs processed:  {len(pdf_files)}")
        print(f"Total chunks created:  {len(chunks)}")
        print(f"Total vectors stored:  {vector_store.total_vectors}")
        print(f"Embedding dimension:   {embedder.embedding_dim}")
        print(f"Index location:        {index_path}")
        print()
        print("Next steps:")
        print("  1. Test retrieval: python scripts/test_retrieval.py")
        print("  2. Launch interface: make run")
        print("  3. Or: streamlit run src/interface/app.py")
        print()

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        logger.info("Build cancelled by user")

    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        print(f"\n❌ Build failed: {e}")
        print("   Check logs for details: data/logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# WORKFLOW:
# 1. Scan data/raw/ pour PDFs
# 2. Pour chaque PDF:
#    - Extraire texte avec PDFProcessor (préserve LaTeX)
#    - Chunker avec Chunker (JAMAIS couper formules)
#    - Ajouter métadonnées (source, page, formules, etc.)
# 3. Générer embeddings (batch processing pour performance)
# 4. Stocker dans FAISS
# 5. Sauvegarder index + métadonnées
#
# VALIDATION CRITIQUE:
# - Après chunking, vérifier que TOUTES les formules LaTeX sont intactes
# - Si une formule est coupée, logger warning
# - Le Chunker lui-même a déjà validation built-in (raise ChunkingError)
#
# PERFORMANCE:
# - Batch embedding (32 chunks à la fois par défaut)
# - Progress bars avec tqdm
# - GPU auto-détecté par Embedder
#
# ERROR HANDLING:
# - Continue si un PDF échoue (log error mais continue)
# - Raise si embedding échoue (critique pour vector store)
# - Validation avant sauvegarde
#
# USAGE:
# ```bash
# # Build normal
# python scripts/build_vector_store.py
#
# # Force rebuild
# python scripts/build_vector_store.py --rebuild
#
# # Custom PDF directory
# python scripts/build_vector_store.py --pdf-dir /path/to/pdfs
#
# # Larger batches (si beaucoup de RAM/GPU)
# python scripts/build_vector_store.py --batch-size 64
# ```
#
# DÉPENDANCES:
# - Tous les modules src/ doivent être implémentés
# - PDFs dans data/raw/
# - Config correctement configurée
#
# OUTPUTS:
# - data/vector_store/default.index (FAISS index binaire)
# - data/vector_store/default.metadata (JSON avec métadonnées)
# - data/logs/app.log (logs détaillés)
#
# ═══════════════════════════════════════════════════════════════════════════════
