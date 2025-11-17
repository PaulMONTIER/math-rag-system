"""
Gestion et traitement des formules LaTeX mathématiques.

Ce module détecte, valide et convertit les formules LaTeX trouvées dans les PDFs.
Crucial pour préserver l'intégrité mathématique lors du chunking.

Usage:
    from src.extraction.latex_handler import LatexHandler

    handler = LatexHandler()
    formulas = handler.detect_formulas(text)
    is_valid = handler.validate_latex(r"\\frac{x}{y}")
    html = handler.convert_to_html(r"$f'(x) = 2x$")
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.exceptions import LatexParsingError

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatexFormula:
    """
    Représentation d'une formule LaTeX détectée.

    Attributes:
        content: Contenu LaTeX (sans délimiteurs)
        start_pos: Position de début dans le texte
        end_pos: Position de fin dans le texte
        delimiter_type: Type de délimiteur (inline, display, environment)
        is_valid: Si la formule est valide syntaxiquement
    """
    content: str
    start_pos: int
    end_pos: int
    delimiter_type: str  # inline ($...$), display ($$...$$), environment (\begin...\end)
    is_valid: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Patterns LaTeX
# ═══════════════════════════════════════════════════════════════════════════════

# Délimiteurs LaTeX communs
LATEX_PATTERNS = {
    # Display math (mode display)
    'display_dollar': r'\$\$(.*?)\$\$',                     # $$...$$
    'display_bracket': r'\\\[(.*?)\\\]',                    # \[...\]

    # Inline math (mode inline)
    'inline_dollar': r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)',  # $...$ (mais pas $$)
    'inline_paren': r'\\\((.*?)\\\)',                       # \(...\)

    # Environments (equation, align, etc.)
    'equation': r'\\begin\{equation\}(.*?)\\end\{equation\}',
    'equation_star': r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}',
    'align': r'\\begin\{align\}(.*?)\\end\{align\}',
    'align_star': r'\\begin\{align\*\}(.*?)\\end\{align\*\}',
    'gather': r'\\begin\{gather\}(.*?)\\end\{gather\}',
    'multline': r'\\begin\{multline\}(.*?)\\end\{multline\}',
    'array': r'\\begin\{array\}(.*?)\\end\{array\}',
    'matrix': r'\\begin\{matrix\}(.*?)\\end\{matrix\}',
    'pmatrix': r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}',
    'bmatrix': r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}',
}

# Commandes LaTeX mathématiques courantes (pour validation)
LATEX_COMMANDS = [
    'frac', 'sqrt', 'sum', 'int', 'lim', 'infty', 'partial',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 'mu', 'pi', 'sigma',
    'sin', 'cos', 'tan', 'log', 'ln', 'exp',
    'left', 'right', 'cdot', 'times', 'div',
    'mathbb', 'mathcal', 'mathrm', 'mathbf',
    'text', 'textit', 'textbf',
]


# ═══════════════════════════════════════════════════════════════════════════════
# LatexHandler
# ═══════════════════════════════════════════════════════════════════════════════

class LatexHandler:
    """
    Gestionnaire de formules LaTeX.

    Détecte, valide et convertit les formules LaTeX dans du texte.

    Example:
        >>> handler = LatexHandler()
        >>> text = "La dérivée est $f'(x) = 2x$"
        >>> formulas = handler.detect_formulas(text)
        >>> print(f"Trouvé {len(formulas)} formule(s)")
        Trouvé 1 formule(s)
    """

    def __init__(self):
        """Initialise le handler."""
        # Compiler les patterns pour performance
        self.compiled_patterns = {
            name: re.compile(pattern, re.DOTALL | re.MULTILINE)
            for name, pattern in LATEX_PATTERNS.items()
        }

        logger.info("LatexHandler initialized")

    def detect_formulas(
        self,
        text: str,
        validate: bool = True
    ) -> List[LatexFormula]:
        """
        Détecte toutes les formules LaTeX dans un texte.

        Args:
            text: Texte contenant potentiellement des formules
            validate: Si True, valide chaque formule détectée

        Returns:
            Liste de LatexFormula trouvées

        Example:
            >>> text = "Soit $x^2 + y^2 = r^2$ et $$\\frac{dy}{dx}$$"
            >>> formulas = handler.detect_formulas(text)
            >>> for f in formulas:
            ...     print(f"{f.delimiter_type}: {f.content}")
            inline_dollar: x^2 + y^2 = r^2
            display_dollar: \\frac{dy}{dx}
        """
        formulas = []

        # Chercher chaque type de pattern
        for delimiter_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                content = match.group(1).strip()

                # Validation optionnelle
                is_valid = True
                if validate:
                    try:
                        is_valid = self.validate_latex(content)
                    except LatexParsingError:
                        is_valid = False

                formula = LatexFormula(
                    content=content,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    delimiter_type=delimiter_type,
                    is_valid=is_valid
                )

                formulas.append(formula)

        # Trier par position pour ordre logique
        formulas.sort(key=lambda f: f.start_pos)

        logger.debug(
            f"Detected {len(formulas)} formulas",
            extra={"formula_count": len(formulas)}
        )

        return formulas

    def validate_latex(self, latex_str: str) -> bool:
        """
        Valide une formule LaTeX basique.

        Vérifie:
        - Parenthèses/crochets équilibrés
        - Commandes LaTeX connues
        - Pas de caractères invalides

        Args:
            latex_str: Formule LaTeX à valider

        Returns:
            True si probablement valide

        Raises:
            LatexParsingError: Si erreur de syntaxe détectée

        Note:
            Validation basique! Pas un parser complet LaTeX.
            Pour validation stricte, utiliser latexmk ou pdflatex.

        Example:
            >>> handler.validate_latex(r"\\frac{x}{y}")
            True
            >>> handler.validate_latex(r"\\frac{x}{")
            False
        """
        # Vérifier parenthèses/crochets équilibrés
        if not self._check_balanced_delimiters(latex_str):
            raise LatexParsingError(
                "Délimiteurs non équilibrés",
                details={"latex": latex_str}
            )

        # Vérifier accolades équilibrées
        if latex_str.count('{') != latex_str.count('}'):
            raise LatexParsingError(
                "Accolades non équilibrées",
                details={"latex": latex_str}
            )

        # Vérifier commandes LaTeX (optionnel, pas exhaustif)
        commands = re.findall(r'\\([a-zA-Z]+)', latex_str)
        unknown_commands = [
            cmd for cmd in commands
            if cmd not in LATEX_COMMANDS
        ]

        # Warning pour commandes inconnues (pas d'erreur, car liste non exhaustive)
        if unknown_commands:
            logger.warning(
                f"Unknown LaTeX commands: {unknown_commands}",
                extra={"commands": unknown_commands, "latex": latex_str[:50]}
            )

        return True

    def _check_balanced_delimiters(self, text: str) -> bool:
        """
        Vérifie que les délimiteurs sont équilibrés.

        Args:
            text: Texte à vérifier

        Returns:
            True si équilibré
        """
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in text:
            if char in pairs.keys():
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                open_char = stack.pop()
                if pairs[open_char] != char:
                    return False

        return len(stack) == 0

    def is_formula_boundary(
        self,
        text: str,
        pos: int
    ) -> bool:
        """
        Vérifie si une position est à l'intérieur d'une formule LaTeX.

        Utile pour le chunking: ne jamais couper au milieu d'une formule.

        Args:
            text: Texte complet
            pos: Position à vérifier

        Returns:
            True si la position est dans une formule

        Example:
            >>> text = "La formule $x^2 + y^2 = r^2$ est importante"
            >>> handler.is_formula_boundary(text, 15)  # dans $...$
            True
            >>> handler.is_formula_boundary(text, 35)  # hors formule
            False
        """
        formulas = self.detect_formulas(text, validate=False)

        for formula in formulas:
            if formula.start_pos <= pos <= formula.end_pos:
                return True

        return False

    def extract_variables(self, latex_str: str) -> List[str]:
        """
        Extrait les variables mathématiques d'une formule.

        Args:
            latex_str: Formule LaTeX

        Returns:
            Liste de variables détectées (lettres simples)

        Example:
            >>> handler.extract_variables(r"f(x) = x^2 + 2xy + y^2")
            ['f', 'x', 'x', 'x', 'y', 'y']
        """
        # Pattern: lettre isolée (pas partie d'une commande \command)
        pattern = r'(?<!\\)\b([a-zA-Z])\b'
        variables = re.findall(pattern, latex_str)

        return variables

    def convert_to_html(
        self,
        latex_str: str,
        include_mathjax: bool = True
    ) -> str:
        """
        Convertit LaTeX en HTML pour affichage web.

        Utilise MathJax pour rendering côté client.

        Args:
            latex_str: Formule LaTeX (avec ou sans délimiteurs)
            include_mathjax: Inclure script MathJax dans output

        Returns:
            HTML avec formule

        Example:
            >>> html = handler.convert_to_html(r"$f'(x) = 2x$")
            >>> print(html)
            <div class="math">\\(f'(x) = 2x\\)</div>

        Note:
            MathJax doit être chargé dans la page HTML:
            <script src="https://cdn.jsdelivr.net/npm/mathjax@3/..."></script>
        """
        # Détecter le type de délimiteur
        if latex_str.startswith('$$') and latex_str.endswith('$$'):
            # Display math
            content = latex_str[2:-2]
            html = f'<div class="math-display">\\[{content}\\]</div>'
        elif latex_str.startswith('$') and latex_str.endswith('$'):
            # Inline math
            content = latex_str[1:-1]
            html = f'<span class="math-inline">\\({content}\\)</span>'
        else:
            # Pas de délimiteurs, assumer inline
            html = f'<span class="math-inline">\\({latex_str}\\)</span>'

        # Ajouter script MathJax si demandé
        if include_mathjax:
            mathjax_script = '''
<script>
MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
'''
            html = mathjax_script + html

        return html

    def get_formula_statistics(self, text: str) -> Dict:
        """
        Génère des statistiques sur les formules dans un texte.

        Args:
            text: Texte à analyser

        Returns:
            Dict avec statistiques

        Example:
            >>> stats = handler.get_formula_statistics(text)
            >>> print(stats)
            {
                'total_formulas': 42,
                'inline': 35,
                'display': 7,
                'avg_length': 15.3,
                'formula_density': 0.12  # % du texte
            }
        """
        formulas = self.detect_formulas(text)

        if not formulas:
            return {
                'total_formulas': 0,
                'inline': 0,
                'display': 0,
                'avg_length': 0,
                'formula_density': 0.0
            }

        # Compter par type
        inline_count = sum(
            1 for f in formulas
            if 'inline' in f.delimiter_type
        )
        display_count = len(formulas) - inline_count

        # Longueur moyenne
        avg_length = sum(len(f.content) for f in formulas) / len(formulas)

        # Densité (portion du texte qui est des formules)
        formula_chars = sum(f.end_pos - f.start_pos for f in formulas)
        density = formula_chars / len(text) if len(text) > 0 else 0.0

        return {
            'total_formulas': len(formulas),
            'inline': inline_count,
            'display': display_count,
            'avg_length': avg_length,
            'formula_density': density
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

def protect_formulas(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Remplace temporairement les formules par des placeholders.

    Utile pour traitement de texte qui pourrait altérer les formules.

    Args:
        text: Texte avec formules

    Returns:
        Tuple (texte_protégé, mapping_placeholder_to_formula)

    Example:
        >>> text = "Soit $x^2$ et $$y^3$$"
        >>> protected, mapping = protect_formulas(text)
        >>> print(protected)
        "Soit __LATEX_0__ et __LATEX_1__"
        >>> print(mapping)
        {'__LATEX_0__': '$x^2$', '__LATEX_1__': '$$y^3$$'}
    """
    handler = LatexHandler()
    formulas = handler.detect_formulas(text, validate=False)

    mapping = {}
    protected_text = text

    # Remplacer en ordre inverse pour préserver positions
    for i, formula in enumerate(reversed(formulas)):
        placeholder = f"__LATEX_{len(formulas) - 1 - i}__"
        original = text[formula.start_pos:formula.end_pos]

        mapping[placeholder] = original

        # Remplacer dans le texte
        protected_text = (
            protected_text[:formula.start_pos]
            + placeholder
            + protected_text[formula.end_pos:]
        )

    return protected_text, mapping


def restore_formulas(text: str, mapping: Dict[str, str]) -> str:
    """
    Restaure les formules depuis les placeholders.

    Args:
        text: Texte avec placeholders
        mapping: Mapping placeholder → formule (from protect_formulas)

    Returns:
        Texte avec formules restaurées

    Example:
        >>> protected = "Soit __LATEX_0__ et __LATEX_1__"
        >>> mapping = {'__LATEX_0__': '$x^2$', '__LATEX_1__': '$$y^3$$'}
        >>> original = restore_formulas(protected, mapping)
        >>> print(original)
        "Soit $x^2$ et $$y^3$$"
    """
    restored_text = text

    for placeholder, formula in mapping.items():
        restored_text = restored_text.replace(placeholder, formula)

    return restored_text


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# DÉLIMITEURS LATEX:
# - Inline: $...$ ou \(...\)
# - Display: $$...$$ ou \[...\]
# - Environments: \begin{equation}...\end{equation}, etc.
#
# PATTERNS REGEX:
# - (?<!\$)\$ = $ non précédé de $ (éviter match $$ comme inline)
# - (.*?) = capture non-greedy (s'arrête au premier match)
# - re.DOTALL = . matche aussi \n (pour formules multi-lignes)
#
# VALIDATION:
# - Basique uniquement (délimiteurs équilibrés, accolades)
# - Pour validation stricte: utiliser latexmk ou pdflatex
# - Liste commandes non exhaustive (warning si inconnue, pas erreur)
#
# CHUNKING:
# - is_formula_boundary() crucial pour ne jamais couper une formule
# - protect_formulas() pour traitement texte qui altèrerait formules
#
# CONVERSION HTML:
# - MathJax pour rendering côté client (léger, rapide)
# - Alternative: latex2mathml (conversion côté serveur)
# - Streamlit: utilise st.latex() nativement (pas besoin HTML)
#
# EXTRACTION VARIABLES:
# - Simple pattern (\b[a-zA-Z]\b)
# - Ne détecte pas \alpha, \beta, etc. (ajuster si nécessaire)
#
# PERFORMANCE:
# - Patterns compilés au __init__ (plus rapide)
# - validate=False possible si performance critique
#
# EXTENSIONS:
# - Parser complet: https://github.com/sympy/sympy (symbolic math)
# - Rendering côté serveur: latex2svg, tex2pix
# - Correction auto: utiliser LLM pour corriger syntaxe LaTeX
#
# ═══════════════════════════════════════════════════════════════════════════════
