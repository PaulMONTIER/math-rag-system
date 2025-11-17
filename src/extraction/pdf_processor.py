"""
Extraction de texte et formules LaTeX depuis des PDFs.

Ce module gère l'extraction optimisée de contenu PDF avec:
- Préservation des formules LaTeX
- Détection de la structure (sections, chapitres)
- Nettoyage du texte
- Extraction de métadonnées

Usage:
    from src.extraction.pdf_processor import PDFProcessor

    processor = PDFProcessor(config)
    doc = processor.process_pdf("path/to/document.pdf")
    print(f"Extrait {len(doc.text)} caractères")
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.exceptions import PDFExtractionError
from src.extraction.latex_handler import LatexHandler

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PDFMetadata:
    """Métadonnées d'un PDF."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    creation_date: Optional[datetime] = None
    page_count: int = 0
    file_size: int = 0  # bytes
    file_path: Optional[Path] = None


@dataclass
class PDFSection:
    """Section d'un PDF (chapitre, partie, etc.)."""
    title: str
    level: int  # 1=chapitre, 2=section, 3=sous-section
    start_page: int
    end_page: Optional[int] = None
    content: str = ""


@dataclass
class PDFDocument:
    """
    Représentation complète d'un document PDF extrait.

    Attributes:
        text: Texte complet extrait
        metadata: Métadonnées du PDF
        pages: Texte par page (dict {page_num: text})
        sections: Sections détectées
        formulas: Formules LaTeX détectées
        statistics: Statistiques d'extraction
    """
    text: str
    metadata: PDFMetadata
    pages: Dict[int, str] = field(default_factory=dict)
    sections: List[PDFSection] = field(default_factory=list)
    formulas: List = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# PDFProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class PDFProcessor:
    """
    Processeur de PDFs optimisé pour documents mathématiques.

    Utilise PyMuPDF (fitz) pour extraction rapide avec préservation LaTeX.

    CHOIX DE BIBLIOTHÈQUE:
    - PyMuPDF (fitz): Rapide, léger, bonne préservation LaTeX
    - Alternative pdfplumber: Meilleur pour tables, plus lent
    - Alternative pymupdf4llm: Wrapper optimisé LLM

    Example:
        >>> config = load_config()
        >>> processor = PDFProcessor(config)
        >>> doc = processor.process_pdf("cours.pdf")
        >>> print(f"{len(doc.formulas)} formulas, {len(doc.text)} chars")
    """

    def __init__(self, config: Optional[object] = None):
        """
        Args:
            config: Objet Config avec pdf_extraction settings
        """
        # Configuration
        if config and hasattr(config, 'pdf_extraction'):
            self.preserve_latex = config.pdf_extraction.preserve_latex
            self.extract_images = config.pdf_extraction.extract_images
            self.ocr_enabled = config.pdf_extraction.ocr_enabled
        else:
            self.preserve_latex = True
            self.extract_images = False
            self.ocr_enabled = False

        # LaTeX handler
        self.latex_handler = LatexHandler()

        logger.info(
            "PDFProcessor initialized",
            extra={
                "library": "PyMuPDF",
                "preserve_latex": self.preserve_latex,
                "extract_images": self.extract_images,
                "ocr_enabled": self.ocr_enabled
            }
        )

    def process_pdf(
        self,
        pdf_path: Path,
        extract_formulas: bool = True,
        detect_sections: bool = True
    ) -> PDFDocument:
        """
        Traite un PDF et extrait tout son contenu.

        Args:
            pdf_path: Chemin vers le PDF
            extract_formulas: Extraire et lister les formules LaTeX
            detect_sections: Détecter la structure (chapitres, sections)

        Returns:
            PDFDocument avec texte, métadonnées, formules, etc.

        Raises:
            PDFExtractionError: Si erreur d'extraction

        Example:
            >>> doc = processor.process_pdf(Path("cours.pdf"))
            >>> print(f"Pages: {doc.metadata.page_count}")
            >>> print(f"Formulas: {len(doc.formulas)}")
        """
        if not pdf_path.exists():
            raise PDFExtractionError(
                f"PDF file not found: {pdf_path}",
                details={"path": str(pdf_path)}
            )

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            # Ouvrir PDF
            pdf_document = fitz.open(pdf_path)

            # Extraire métadonnées
            metadata = self._extract_metadata(pdf_document, pdf_path)

            # Extraire texte page par page
            pages_text = {}
            all_text = []

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_text = self._extract_page_text(page, page_num)

                pages_text[page_num + 1] = page_text  # Pages 1-indexed
                all_text.append(page_text)

            # Texte complet
            full_text = "\n\n".join(all_text)

            # Nettoyage
            full_text = self._clean_text(full_text)

            # Extraire formules si demandé
            formulas = []
            if extract_formulas and self.preserve_latex:
                formulas = self.latex_handler.detect_formulas(full_text)
                logger.info(f"Extracted {len(formulas)} formulas")

            # Détecter sections si demandé
            sections = []
            if detect_sections:
                sections = self._detect_sections(pdf_document, pages_text)
                logger.info(f"Detected {len(sections)} sections")

            # Statistiques
            stats = self._compute_statistics(full_text, formulas, pages_text)

            # Fermer PDF
            pdf_document.close()

            # Créer document
            doc = PDFDocument(
                text=full_text,
                metadata=metadata,
                pages=pages_text,
                sections=sections,
                formulas=formulas,
                statistics=stats
            )

            logger.info(
                f"PDF processed successfully: {pdf_path.name}",
                extra={
                    "pages": metadata.page_count,
                    "chars": len(full_text),
                    "formulas": len(formulas),
                    "sections": len(sections)
                }
            )

            return doc

        except Exception as e:
            raise PDFExtractionError(
                f"Failed to extract PDF {pdf_path.name}: {e}",
                details={"path": str(pdf_path), "error": str(e)}
            ) from e

    def _extract_metadata(
        self,
        pdf_document: fitz.Document,
        pdf_path: Path
    ) -> PDFMetadata:
        """
        Extrait les métadonnées du PDF.

        Args:
            pdf_document: Document PyMuPDF
            pdf_path: Chemin du fichier

        Returns:
            PDFMetadata
        """
        meta_dict = pdf_document.metadata

        # Parser date si présente
        creation_date = None
        if meta_dict.get('creationDate'):
            try:
                # Format PyMuPDF: D:20240101120000
                date_str = meta_dict['creationDate'].replace('D:', '')
                creation_date = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
            except (ValueError, KeyError):
                pass

        # Keywords (split par virgule ou point-virgule)
        keywords = None
        if meta_dict.get('keywords'):
            keywords = re.split(r'[;,]\s*', meta_dict['keywords'])

        return PDFMetadata(
            title=meta_dict.get('title'),
            author=meta_dict.get('author'),
            subject=meta_dict.get('subject'),
            keywords=keywords,
            creation_date=creation_date,
            page_count=pdf_document.page_count,
            file_size=pdf_path.stat().st_size,
            file_path=pdf_path
        )

    def _extract_page_text(
        self,
        page: fitz.Page,
        page_num: int
    ) -> str:
        """
        Extrait le texte d'une page avec préservation LaTeX.

        Args:
            page: Page PyMuPDF
            page_num: Numéro de page (0-indexed)

        Returns:
            Texte de la page

        Note:
            PyMuPDF préserve généralement bien les formules LaTeX
            si le PDF a été généré depuis LaTeX.
        """
        try:
            # Extraction avec options optimisées
            text = page.get_text(
                "text",  # Format texte simple
                sort=True  # Trier par position sur la page
            )

            # Si OCR activé et page vide (PDF scanné)
            if self.ocr_enabled and not text.strip():
                text = self._ocr_page(page)

            return text

        except Exception as e:
            logger.warning(
                f"Error extracting page {page_num}: {e}",
                extra={"page": page_num}
            )
            return ""

    def _ocr_page(self, page: fitz.Page) -> str:
        """
        OCR d'une page (PDFs scannés).

        Args:
            page: Page PyMuPDF

        Returns:
            Texte extrait par OCR

        Note:
            Nécessite pytesseract installé.
            Pas implémenté par défaut (retourne chaîne vide).
        """
        logger.warning("OCR not implemented yet (requires pytesseract)")
        return ""

        # TODO: Implémenter OCR
        # try:
        #     import pytesseract
        #     from PIL import Image
        #     pix = page.get_pixmap()
        #     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        #     text = pytesseract.image_to_string(img, lang='fra+eng')
        #     return text
        # except ImportError:
        #     logger.error("pytesseract not installed")
        #     return ""

    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte extrait.

        Opérations:
        - Normaliser espaces multiples
        - Retirer caractères de contrôle
        - Préserver sauts de ligne de paragraphes
        - Préserver formules LaTeX intactes

        Args:
            text: Texte brut

        Returns:
            Texte nettoyé
        """
        # Protéger formules LaTeX pendant nettoyage
        from src.extraction.latex_handler import protect_formulas, restore_formulas
        protected, mapping = protect_formulas(text)

        # Retirer caractères de contrôle (sauf \n, \t)
        protected = ''.join(
            char for char in protected
            if char.isprintable() or char in ['\n', '\t']
        )

        # Normaliser espaces horizontaux (mais pas \n)
        protected = re.sub(r'[ \t]+', ' ', protected)

        # Normaliser sauts de ligne (max 2 consécutifs = paragraphe)
        protected = re.sub(r'\n{3,}', '\n\n', protected)

        # Restaurer formules
        cleaned = restore_formulas(protected, mapping)

        return cleaned.strip()

    def _detect_sections(
        self,
        pdf_document: fitz.Document,
        pages_text: Dict[int, str]
    ) -> List[PDFSection]:
        """
        Détecte la structure du document (chapitres, sections).

        Utilise:
        - Table des matières (ToC) si présente dans PDF
        - Patterns de titres (fallback)

        Args:
            pdf_document: Document PyMuPDF
            pages_text: Texte par page

        Returns:
            Liste de PDFSection

        Note:
            Pas toujours précis (dépend de la qualité du PDF).
        """
        sections = []

        # Essayer ToC embarquée
        try:
            toc = pdf_document.get_toc()  # [[level, title, page], ...]

            for level, title, page in toc:
                section = PDFSection(
                    title=title.strip(),
                    level=level,
                    start_page=page
                )
                sections.append(section)

            if sections:
                logger.info(f"Extracted {len(sections)} sections from ToC")
                return sections

        except Exception as e:
            logger.debug(f"No ToC found or error: {e}")

        # Fallback: détecter patterns de titres
        sections = self._detect_sections_by_pattern(pages_text)

        return sections

    def _detect_sections_by_pattern(
        self,
        pages_text: Dict[int, str]
    ) -> List[PDFSection]:
        """
        Détecte sections par patterns de texte.

        Cherche:
        - Chapitre X, Chapter X
        - Section X.Y
        - Lignes en majuscules
        - Lignes commençant par numéros (1., 1.1, etc.)

        Args:
            pages_text: Texte par page

        Returns:
            Liste de PDFSection
        """
        sections = []

        # Patterns de titres
        patterns = {
            1: [
                r'^(CHAPITRE|CHAPTER)\s+(\d+|[IVX]+)[\s:.-]*(.*?)$',
                r'^(PARTIE|PART)\s+(\d+|[IVX]+)[\s:.-]*(.*?)$',
            ],
            2: [
                r'^(\d+)\.\s+(.*?)$',                      # 1. Titre
                r'^(Section|SECTION)\s+(\d+)[\s:.-]*(.*?)$',
            ],
            3: [
                r'^(\d+)\.(\d+)\s+(.*?)$',                 # 1.1 Titre
            ],
        }

        for page_num, text in pages_text.items():
            lines = text.split('\n')

            for line in lines:
                line = line.strip()

                if not line or len(line) < 5:
                    continue

                # Tester chaque pattern
                for level, level_patterns in patterns.items():
                    for pattern in level_patterns:
                        match = re.match(pattern, line, re.IGNORECASE)

                        if match:
                            # Extraire titre
                            title = match.group(0).strip()

                            section = PDFSection(
                                title=title,
                                level=level,
                                start_page=page_num
                            )
                            sections.append(section)
                            break

        logger.debug(f"Detected {len(sections)} sections by pattern")
        return sections

    def _compute_statistics(
        self,
        text: str,
        formulas: List,
        pages_text: Dict[int, str]
    ) -> Dict:
        """
        Calcule des statistiques sur le document extrait.

        Args:
            text: Texte complet
            formulas: Formules détectées
            pages_text: Texte par page

        Returns:
            Dict avec statistiques
        """
        # Mots (approximation)
        words = text.split()

        # Statistiques formules
        formula_stats = self.latex_handler.get_formula_statistics(text)

        # Longueur moyenne des pages
        page_lengths = [len(p) for p in pages_text.values()]
        avg_page_length = sum(page_lengths) / len(page_lengths) if page_lengths else 0

        return {
            'total_chars': len(text),
            'total_words': len(words),
            'total_pages': len(pages_text),
            'avg_page_length': avg_page_length,
            'formula_count': len(formulas),
            'formula_stats': formula_stats,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def batch_process_pdfs(
    pdf_paths: List[Path],
    processor: PDFProcessor,
    save_dir: Optional[Path] = None
) -> List[PDFDocument]:
    """
    Traite un batch de PDFs.

    Args:
        pdf_paths: Liste de chemins PDFs
        processor: PDFProcessor configuré
        save_dir: Répertoire de sauvegarde (optionnel)

    Returns:
        Liste de PDFDocument

    Example:
        >>> pdfs = list(Path("data/raw").glob("*.pdf"))
        >>> processor = PDFProcessor(config)
        >>> docs = batch_process_pdfs(pdfs, processor)
        >>> print(f"Processed {len(docs)} documents")
    """
    documents = []

    for i, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"Processing {i}/{len(pdf_paths)}: {pdf_path.name}")

        try:
            doc = processor.process_pdf(pdf_path)
            documents.append(doc)

            # Sauvegarder si demandé
            if save_dir:
                save_document(doc, save_dir)

        except PDFExtractionError as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    logger.info(f"Batch processing complete: {len(documents)}/{len(pdf_paths)} successful")

    return documents


def save_document(doc: PDFDocument, output_dir: Path) -> Path:
    """
    Sauvegarde un document extrait en JSON.

    Args:
        doc: PDFDocument à sauvegarder
        output_dir: Répertoire de sortie

    Returns:
        Chemin du fichier sauvegardé
    """
    import json
    from dataclasses import asdict

    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier
    filename = doc.metadata.file_path.stem + ".json"
    output_path = output_dir / filename

    # Convertir en dict (avec gestion dates)
    doc_dict = {
        'text': doc.text,
        'metadata': {
            **asdict(doc.metadata),
            'creation_date': doc.metadata.creation_date.isoformat() if doc.metadata.creation_date else None,
            'file_path': str(doc.metadata.file_path)
        },
        'pages': doc.pages,
        'sections': [asdict(s) for s in doc.sections],
        'statistics': doc.statistics
    }

    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Document saved: {output_path}")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# CHOIX BIBLIOTHÈQUE:
# PyMuPDF (fitz) choisi car:
#   ✓ Rapide (C++, wrappé Python)
#   ✓ Léger (~15MB vs ~100MB pdfplumber)
#   ✓ Bonne préservation LaTeX (si PDF généré depuis LaTeX)
#   ✓ Extraction structurée (ToC, métadonnées)
#   ✓ Support images/diagrammes
#
# Alternative pdfplumber si:
#   - Besoin extraction précise de tables
#   - PDFs complexes avec layouts multiples
#   - Extraction coordonnées exactes
#
# PRÉSERVATION LATEX:
# - Fonctionne bien si PDF généré depuis .tex
# - Peut échouer si PDF créé autrement (Word, etc.)
# - Formules alors converties en Unicode (∫, ∑, √)
# - Fallback: détecter Unicode math et reconstruire LaTeX (complexe)
#
# OCR:
# - Nécessaire pour PDFs scannés (images de pages)
# - Nécessite pytesseract + Tesseract installé système
# - Ajouter: pip install pytesseract
# - Installer Tesseract: brew/apt-get/chocolatey install tesseract
#
# DÉTECTION STRUCTURE:
# - ToC embarquée: Meilleur (si présente)
# - Patterns: Heuristique, pas toujours précis
# - Amélioration possible: utiliser positions/tailles de texte
#
# NETTOYAGE:
# - protect_formulas() crucial pour ne pas altérer LaTeX
# - Normalisation espaces sans casser formules
# - Préserver structure paragraphes (double \n)
#
# PERFORMANCE:
# - PyMuPDF très rapide (~10-50 pages/sec selon complexité)
# - OCR beaucoup plus lent (~1 page/sec)
# - Batch processing avec multiprocessing possible
#
# EXTENSIONS:
# - Extraction images/diagrammes (pour multimodal)
# - Extraction tables (pdfplumber ou camelot)
# - Reconstruction LaTeX depuis Unicode
# - Détection langue (langdetect)
# - Parsing références bibliographiques
#
# ═══════════════════════════════════════════════════════════════════════════════
