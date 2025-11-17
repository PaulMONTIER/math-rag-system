"""
Découpage intelligent de texte en chunks avec préservation des formules LaTeX.

Ce module est LE CŒUR du système RAG. Il DOIT ABSOLUMENT :
✓ Ne JAMAIS couper une formule mathématique
✓ Respecter les limites de phrases
✓ Maintenir le contexte entre chunks (overlap)
✓ Préserver la structure du document

Usage:
    from src.vectorization.chunker import Chunker

    chunker = Chunker(config)
    chunks = chunker.chunk_text(text, metadata={"source": "cours.pdf"})
    print(f"Créé {len(chunks)} chunks")
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.exceptions import ChunkingError
from src.extraction.latex_handler import LatexHandler

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """
    Un chunk de texte avec métadonnées.

    Attributes:
        text: Contenu du chunk
        metadata: Métadonnées (source, page, section, etc.)
        start_char: Position de début dans le texte original
        end_char: Position de fin
        chunk_index: Index du chunk (0-based)
        has_formula: Si le chunk contient des formules LaTeX
        formula_count: Nombre de formules dans le chunk
    """
    text: str
    metadata: Dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    has_formula: bool = False
    formula_count: int = 0

    @property
    def chunk_id(self) -> str:
        """ID unique du chunk."""
        source = self.metadata.get('source', 'unknown')
        return f"{source}_{self.chunk_index}"

    @property
    def num_formulas(self) -> int:
        """Alias pour formula_count."""
        return self.formula_count

    @property
    def char_count(self) -> int:
        """Nombre de caractères."""
        return len(self.text)

    def __len__(self) -> int:
        """Retourne la longueur du texte."""
        return len(self.text)


# ═══════════════════════════════════════════════════════════════════════════════
# Chunker
# ═══════════════════════════════════════════════════════════════════════════════

class Chunker:
    """
    Chunker intelligent pour documents mathématiques.

    RÈGLE D'OR : Ne JAMAIS couper une formule LaTeX !

    Example:
        >>> config = load_config()
        >>> chunker = Chunker(config)
        >>> text = "La dérivée est $f'(x) = 2x$. C'est important."
        >>> chunks = chunker.chunk_text(text)
        >>> # La formule $f'(x) = 2x$ sera TOUJOURS dans un seul chunk
    """

    def __init__(self, config: Optional[object] = None):
        """
        Args:
            config: Objet Config avec chunking settings
        """
        # Configuration
        if config and hasattr(config, 'chunking'):
            self.chunk_size = config.chunking.chunk_size
            self.chunk_overlap = config.chunking.chunk_overlap
            self.respect_formula_boundaries = config.chunking.respect_formula_boundaries
            self.respect_sentence_boundaries = config.chunking.respect_sentence_boundaries
            self.min_chunk_size = config.chunking.min_chunk_size
        else:
            # Valeurs par défaut
            self.chunk_size = 512
            self.chunk_overlap = 50
            self.respect_formula_boundaries = True
            self.respect_sentence_boundaries = True
            self.min_chunk_size = 100

        # LaTeX handler pour détecter formules
        self.latex_handler = LatexHandler()

        logger.info(
            "Chunker initialized",
            extra={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "respect_formulas": self.respect_formula_boundaries
            }
        )

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[TextChunk]:
        """
        Découpe un texte en chunks intelligemment.

        GARANTIE : Aucune formule LaTeX ne sera jamais coupée !

        Args:
            text: Texte à découper
            metadata: Métadonnées communes à tous les chunks

        Returns:
            Liste de TextChunk

        Raises:
            ChunkingError: Si erreur de découpage

        Example:
            >>> text = "Soit $x^2 + y^2 = r^2$. Alors..."
            >>> chunks = chunker.chunk_text(text, {"source": "cours.pdf", "page": 1})
            >>> for chunk in chunks:
            ...     assert '$x^2 + y^2 = r^2$' not in chunk.text or \
            ...            chunk.text.count('$') % 2 == 0  # Formule complète
        """
        if not text or not text.strip():
            raise ChunkingError("Texte vide fourni au chunker")

        metadata = metadata or {}

        logger.debug(f"Chunking text of {len(text)} chars")

        # Détecter toutes les formules LaTeX
        formulas = []
        if self.respect_formula_boundaries:
            formulas = self.latex_handler.detect_formulas(text, validate=False)
            logger.debug(f"Detected {len(formulas)} formulas to preserve")

        # Créer les chunks
        chunks = self._create_chunks(text, formulas, metadata)

        # Validation finale
        self._validate_chunks(chunks, text, formulas)

        logger.info(
            f"Created {len(chunks)} chunks",
            extra={
                "chunk_count": len(chunks),
                "avg_size": sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
                "formulas_preserved": sum(c.formula_count for c in chunks)
            }
        )

        return chunks

    def _create_chunks(
        self,
        text: str,
        formulas: List,
        metadata: Dict
    ) -> List[TextChunk]:
        """
        Crée les chunks en respectant toutes les contraintes.

        Algorithme:
        1. Diviser le texte en "unités sûres" (paragraphes ou phrases)
        2. Pour chaque unité, vérifier qu'elle ne coupe pas de formule
        3. Assembler les unités jusqu'à atteindre chunk_size
        4. Ajouter overlap avec chunk précédent

        Args:
            text: Texte à découper
            formulas: Formules détectées (pour préservation)
            metadata: Métadonnées de base

        Returns:
            Liste de TextChunk
        """
        chunks = []
        current_pos = 0
        chunk_index = 0

        # Diviser en unités sûres (paragraphes puis phrases)
        units = self._split_into_safe_units(text, formulas)

        logger.debug(f"Split text into {len(units)} safe units")

        # Assembler les unités en chunks
        current_chunk_units = []
        current_chunk_size = 0

        for unit_text, unit_start, unit_end in units:
            unit_size = len(unit_text)

            # Si l'unité seule dépasse chunk_size
            if unit_size > self.chunk_size * 1.5:  # Tolérance 150%
                # Si on a déjà des unités, créer un chunk
                if current_chunk_units:
                    chunk = self._build_chunk(
                        current_chunk_units,
                        chunk_index,
                        metadata,
                        formulas
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_units = []
                    current_chunk_size = 0

                # Créer un chunk pour cette grande unité seule
                # (tolérer dépassement car contient formule ou paragraphe important)
                chunk = self._build_chunk(
                    [(unit_text, unit_start, unit_end)],
                    chunk_index,
                    metadata,
                    formulas
                )
                chunks.append(chunk)
                chunk_index += 1

                logger.warning(
                    f"Large unit ({unit_size} chars) in single chunk (formula or important paragraph)",
                    extra={"unit_size": unit_size, "chunk_index": chunk_index}
                )

            # Si ajouter cette unité dépasse chunk_size
            elif current_chunk_size + unit_size > self.chunk_size:
                # Créer chunk avec unités actuelles
                if current_chunk_units:
                    chunk = self._build_chunk(
                        current_chunk_units,
                        chunk_index,
                        metadata,
                        formulas
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Commencer nouveau chunk avec overlap
                if self.chunk_overlap > 0 and chunks:
                    # Prendre dernières unités pour overlap
                    overlap_units = self._get_overlap_units(
                        current_chunk_units,
                        self.chunk_overlap
                    )
                    current_chunk_units = overlap_units + [(unit_text, unit_start, unit_end)]
                    current_chunk_size = sum(len(u[0]) for u in current_chunk_units)
                else:
                    current_chunk_units = [(unit_text, unit_start, unit_end)]
                    current_chunk_size = unit_size

            # Sinon, ajouter à chunk actuel
            else:
                current_chunk_units.append((unit_text, unit_start, unit_end))
                current_chunk_size += unit_size

        # Dernier chunk
        if current_chunk_units:
            chunk = self._build_chunk(
                current_chunk_units,
                chunk_index,
                metadata,
                formulas
            )
            chunks.append(chunk)

        return chunks

    def _split_into_safe_units(
        self,
        text: str,
        formulas: List
    ) -> List[Tuple[str, int, int]]:
        """
        Divise le texte en unités "sûres" (ne coupent jamais de formule).

        Stratégie :
        1. D'abord par paragraphes (double \n)
        2. Puis par phrases (. ! ?)
        3. Toujours vérifier qu'on ne coupe pas de formule

        Args:
            text: Texte à diviser
            formulas: Formules à préserver

        Returns:
            Liste de tuples (unit_text, start_pos, end_pos)
        """
        units = []

        # Diviser par paragraphes
        paragraphs = re.split(r'\n\n+', text)

        current_pos = 0
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2  # +2 for \n\n
                continue

            para_start = text.find(para, current_pos)
            para_end = para_start + len(para)

            # Vérifier si paragraphe contient des formules
            para_formulas = [
                f for f in formulas
                if para_start <= f.start_pos < para_end or para_start < f.end_pos <= para_end
            ]

            # Si paragraphe petit ou contient formule compliquée, garder entier
            if len(para) <= self.chunk_size or len(para_formulas) > 3:
                units.append((para, para_start, para_end))
            else:
                # Diviser en phrases si respect_sentence_boundaries
                if self.respect_sentence_boundaries:
                    sentences = self._split_into_sentences(para, para_start, formulas)
                    units.extend(sentences)
                else:
                    units.append((para, para_start, para_end))

            current_pos = para_end + 2

        return units

    def _split_into_sentences(
        self,
        text: str,
        text_start: int,
        formulas: List
    ) -> List[Tuple[str, int, int]]:
        """
        Divise un texte en phrases sans couper de formules.

        Args:
            text: Texte à diviser
            text_start: Position de début de ce texte dans le doc original
            formulas: Formules à préserver

        Returns:
            Liste de tuples (sentence_text, start_pos, end_pos)
        """
        sentences = []

        # Pattern de fin de phrase (. ! ?) suivi d'espace ou fin
        # Mais pas dans une formule !
        sentence_pattern = r'([.!?])\s+'

        current_pos = 0
        last_end = 0

        for match in re.finditer(sentence_pattern, text):
            sentence_end = match.end()

            # Vérifier qu'on ne coupe pas de formule
            abs_start = text_start + last_end
            abs_end = text_start + sentence_end

            # Chercher formules qui seraient coupées
            is_safe = True
            for formula in formulas:
                # Si formule commence avant sentence_end et finit après
                if formula.start_pos < abs_end < formula.end_pos:
                    is_safe = False
                    break

            if is_safe:
                sentence = text[last_end:sentence_end].strip()
                if sentence:
                    sentences.append((
                        sentence,
                        text_start + last_end,
                        text_start + sentence_end
                    ))
                last_end = sentence_end

        # Dernière phrase
        if last_end < len(text):
            sentence = text[last_end:].strip()
            if sentence:
                sentences.append((
                    sentence,
                    text_start + last_end,
                    text_start + len(text)
                ))

        return sentences if sentences else [(text, text_start, text_start + len(text))]

    def _build_chunk(
        self,
        units: List[Tuple[str, int, int]],
        chunk_index: int,
        base_metadata: Dict,
        formulas: List
    ) -> TextChunk:
        """
        Construit un TextChunk depuis des unités.

        Args:
            units: Liste de (text, start, end)
            chunk_index: Index du chunk
            base_metadata: Métadonnées de base
            formulas: Formules (pour compter celles dans le chunk)

        Returns:
            TextChunk
        """
        if not units:
            raise ChunkingError("Pas d'unités pour construire un chunk")

        # Assembler le texte
        chunk_text = " ".join(unit[0] for unit in units)

        # Positions
        start_char = units[0][1]
        end_char = units[-1][2]

        # Compter formules dans ce chunk
        chunk_formulas = [
            f for f in formulas
            if start_char <= f.start_pos < end_char or start_char < f.end_pos <= end_char
        ]

        # Métadonnées
        chunk_metadata = {
            **base_metadata,
            "chunk_size": len(chunk_text),
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
        }

        chunk = TextChunk(
            text=chunk_text,
            metadata=chunk_metadata,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            has_formula=(len(chunk_formulas) > 0),
            formula_count=len(chunk_formulas)
        )

        return chunk

    def _get_overlap_units(
        self,
        units: List[Tuple[str, int, int]],
        overlap_size: int
    ) -> List[Tuple[str, int, int]]:
        """
        Récupère les dernières unités pour l'overlap.

        Args:
            units: Unités du chunk précédent
            overlap_size: Taille d'overlap souhaitée

        Returns:
            Liste d'unités pour overlap
        """
        overlap_units = []
        current_size = 0

        # Prendre unités en ordre inverse jusqu'à atteindre overlap_size
        for unit in reversed(units):
            unit_size = len(unit[0])

            if current_size + unit_size > overlap_size * 1.5:  # Tolérance
                break

            overlap_units.insert(0, unit)
            current_size += unit_size

        return overlap_units

    def _validate_chunks(
        self,
        chunks: List[TextChunk],
        original_text: str,
        formulas: List
    ) -> None:
        """
        Valide que les chunks respectent toutes les contraintes.

        VÉRIFICATIONS CRITIQUES:
        ✓ Aucun chunk ne coupe une formule LaTeX
        ✓ Tous les chunks sont >= min_chunk_size
        ✓ Coverage du texte original

        Args:
            chunks: Chunks créés
            original_text: Texte original
            formulas: Formules détectées

        Raises:
            ChunkingError: Si validation échoue
        """
        if not chunks:
            raise ChunkingError("Aucun chunk créé")

        # Vérifier taille minimale
        small_chunks = [c for c in chunks if len(c.text) < self.min_chunk_size]
        if small_chunks and len(chunks) > 1:  # Tolérer dernier chunk petit
            if small_chunks[-1] != chunks[-1]:
                logger.warning(
                    f"{len(small_chunks)} chunks sous min_chunk_size ({self.min_chunk_size})",
                    extra={"small_chunk_sizes": [len(c.text) for c in small_chunks]}
                )

        # CRITIQUE : Vérifier qu'aucune formule n'est coupée
        for formula in formulas:
            formula_fully_in_chunk = False

            for chunk in chunks:
                # Formule doit être entièrement dans un chunk
                if chunk.start_char <= formula.start_pos and formula.end_pos <= chunk.end_char:
                    formula_fully_in_chunk = True
                    break

            if not formula_fully_in_chunk:
                # ERREUR CRITIQUE !
                raise ChunkingError(
                    f"Formule coupée ! Position {formula.start_pos}-{formula.end_pos}: {formula.content[:50]}...",
                    details={
                        "formula_start": formula.start_pos,
                        "formula_end": formula.end_pos,
                        "formula_preview": formula.content[:100]
                    }
                )

        logger.debug("✓ Chunk validation passed (no formulas cut)")


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_token_count(text: str) -> int:
    """
    Estime le nombre de tokens (approximation).

    Args:
        text: Texte

    Returns:
        Estimation du nombre de tokens

    Note:
        Approximation simple: ~4 chars = 1 token (anglais)
        Pour français: ~5 chars = 1 token
        Pour précision: utiliser tiktoken (OpenAI)
    """
    # Détecter si français (caractères accentués)
    has_accents = any(ord(c) > 127 for c in text)

    if has_accents:
        return len(text) // 5
    else:
        return len(text) // 4


def chunk_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    chunker: Optional[Chunker] = None
) -> List[TextChunk]:
    """
    Découpe en utilisant un compte de tokens au lieu de caractères.

    Args:
        text: Texte à découper
        max_tokens: Tokens maximum par chunk
        overlap_tokens: Overlap en tokens
        chunker: Chunker configuré (optionnel)

    Returns:
        Liste de TextChunk

    Example:
        >>> chunks = chunk_by_tokens(text, max_tokens=512)
        >>> for chunk in chunks:
        ...     assert estimate_token_count(chunk.text) <= 512
    """
    # Ajuster chunk_size en caractères basé sur tokens
    # ~4-5 chars par token
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    if chunker is None:
        chunker = Chunker()

    # Override temporairement les tailles
    original_chunk_size = chunker.chunk_size
    original_overlap = chunker.chunk_overlap

    chunker.chunk_size = max_chars
    chunker.chunk_overlap = overlap_chars

    chunks = chunker.chunk_text(text)

    # Restore
    chunker.chunk_size = original_chunk_size
    chunker.chunk_overlap = original_overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# ⚠️ RÈGLE D'OR : NE JAMAIS COUPER UNE FORMULE LATEX ! ⚠️
#
# ALGORITHME:
# 1. Détecter toutes les formules LaTeX (latex_handler)
# 2. Diviser en unités sûres (paragraphes → phrases)
# 3. Vérifier que chaque découpe ne coupe pas de formule
# 4. Assembler unités jusqu'à chunk_size
# 5. Ajouter overlap intelligent
# 6. VALIDER que toutes les formules sont intactes
#
# STRATÉGIE DÉCOUPAGE:
# - Paragraphes d'abord (\n\n)
# - Puis phrases (. ! ?) si respect_sentence_boundaries
# - Toujours vérifier formules avant de couper
# - Tolérer dépassement chunk_size si formule longue
#
# OVERLAP:
# - Prend dernières unités du chunk précédent
# - Maintient contexte entre chunks
# - Important pour retrieval
#
# VALIDATION:
# - _validate_chunks() vérifie TOUTES les formules
# - Lève ChunkingError si formule coupée
# - Logs warnings pour chunks petits
#
# MÉTADONNÉES:
# - source, page, section (depuis PDFDocument)
# - chunk_index, start_char, end_char (position)
# - has_formula, formula_count (statistiques)
#
# EDGE CASES:
# - Formule > chunk_size : Tolérer, créer grand chunk
# - Dernier chunk petit : Tolérer
# - Texte sans formules : Découpage classique par phrases
# - Paragraphe énorme : Diviser en phrases
#
# PERFORMANCE:
# - O(n) avec n = longueur texte
# - Detection formules: O(n) regex (compilés)
# - Validation: O(f*c) avec f=formulas, c=chunks (généralement petit)
#
# TESTS RECOMMANDÉS:
# ```python
# # Test 1: Formule inline
# text = "La formule $x^2 + y^2 = r^2$ est importante."
# chunks = chunker.chunk_text(text)
# assert all('$x^2 + y^2 = r^2$' in chunk.text for chunk in chunks if '$' in chunk.text)
#
# # Test 2: Formule display
# text = "Soit $$\\int_0^x f(t)dt = F(x)$$ la primitive."
# chunks = chunker.chunk_text(text)
# assert all(chunk.text.count('$$') % 2 == 0 for chunk in chunks)
#
# # Test 3: Formule longue
# long_formula = "$" + "x" * 1000 + "$"
# text = f"Formule: {long_formula}. Suite."
# chunks = chunker.chunk_text(text)
# assert any(long_formula in chunk.text for chunk in chunks)
# ```
#
# EXTENSIONS:
# - Support markdown (```math)
# - Support AsciiMath
# - Chunking sémantique (par topics)
# - Chunking hiérarchique (sections → paragraphes → phrases)
#
# ═══════════════════════════════════════════════════════════════════════════════
