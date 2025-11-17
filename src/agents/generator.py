"""
Agent Generator - Génération de réponses pédagogiques avec LLM.

Cet agent génère des réponses claires et pédagogiques en utilisant:
- Le contexte récupéré (from retriever)
- Un prompt système optimisé
- Un LLM fermé (GPT ou Claude)

Usage:
    from src.agents.generator import GeneratorAgent

    generator = GeneratorAgent(config, llm_client)
    response = generator.generate(
        question="Qu'est-ce qu'une dérivée ?",
        context="...",
        student_level="L2"
    )
"""

from typing import Optional, Dict, List
from dataclasses import dataclass

from src.llm.closed_models import BaseLLMClient, LLMResponse
from src.agents.retriever import RetrievalResult
from src.utils.logger import get_logger, log_performance
from src.utils.exceptions import AgentError

logger = get_logger(__name__)


@dataclass
class GeneratedResponse:
    """
    Réponse générée par l'agent.

    Attributes:
        answer: Réponse générée
        sources_cited: Sources citées dans la réponse
        llm_response: Réponse brute du LLM (tokens, coût, etc.)
        metadata: Métadonnées additionnelles
    """
    answer: str
    sources_cited: List[str]
    llm_response: LLMResponse
    metadata: Dict


class GeneratorAgent:
    """
    Agent de génération de réponses pédagogiques.

    Responsabilités:
    1. Construire prompt avec contexte
    2. Adapter niveau au profil étudiant
    3. Générer réponse avec LLM
    4. Extraire sources citées
    5. Préserver formules LaTeX

    Example:
        >>> generator = GeneratorAgent(config, llm_client)
        >>> response = generator.generate(
        ...     question="dérivée de x²",
        ...     context=context,
        ...     student_level="L2"
        ... )
        >>> print(response.answer)
    """

    def __init__(
        self,
        config: object,
        llm_client: BaseLLMClient
    ):
        """
        Args:
            config: Objet Config
            llm_client: Client LLM (OpenAI, Anthropic, Ollama)
        """
        self.config = config
        self.llm_client = llm_client

        # Template de prompt système
        if hasattr(config.agents, 'generator'):
            self.system_prompt_template = config.agents.generator.get(
                "system_prompt_template",
                self._get_default_system_prompt()
            )
        else:
            self.system_prompt_template = self._get_default_system_prompt()

        logger.info("GeneratorAgent initialized")

    def _get_default_system_prompt(self) -> str:
        """
        Retourne le prompt système par défaut.

        Returns:
            Template de prompt (avec placeholders {level}, {context}, etc.)
        """
        return """Tu es un assistant pédagogique spécialisé en mathématiques.

Ta mission est de répondre aux questions mathématiques de manière claire, structurée et pédagogique.

NIVEAU DE DÉTAIL: {level}
- "Simple": Explications accessibles, vocabulaire simple, exemples concrets
- "Détaillé": Explications rigoureuses avec justifications, exemples variés
- "Beaucoup de détails": Explications complètes et formelles, démonstrations détaillées

RÈGLES IMPORTANTES:
1. Utilise UNIQUEMENT le contexte fourni pour répondre
2. Le contexte peut contenir:
   - Documents locaux (PDFs de cours) → Cite avec nom de fichier et page
   - Résultats de recherche web (pour actualités/événements récents) → Cite avec titre et URL
3. IMPORTANT: Si tu vois des résultats web dans le contexte (avec URLs), UTILISE-LES pour répondre aux questions d'actualité
4. Adapte ton niveau de détail selon: {level}
5. Donne des exemples concrets quand c'est pertinent
6. Si l'information n'est PAS dans le contexte fourni (ni local ni web), dis-le clairement
7. Ne JAMAIS inventer d'informations ou halluciner
8. Structure ta réponse: définition, explication, exemple (si applicable)

FORMULES MATHÉMATIQUES - TRÈS IMPORTANT:
- Utilise TOUJOURS la notation LaTeX pour les formules mathématiques
- Formules inline (dans le texte): utilise $...$
  Exemple: "La dérivée de $f(x) = x^2$ est $f'(x) = 2x$"
- Formules display (centrées, sur une ligne séparée): utilise $$...$$
  Exemple de limite: $$f'(a) = \\lim_{{h \\to 0}} \\frac{{f(a+h) - f(a)}}{{h}}$$
  Exemple d'intégrale: $$\\int_a^b f(x)dx = F(b) - F(a)$$
- Pour les symboles spéciaux, utilise la notation LaTeX standard: \\lim, \\frac, \\int, \\sum, etc.

Format de réponse attendu:
- Introduction/Définition
- Explication adaptée au niveau de détail ({level})
- Exemple concret avec formules LaTeX (si pertinent)
- Formules mathématiques en notation LaTeX
- Sources citées:
  * Pour documents locaux: nom de fichier et page (ex: "Analyse_L2.pdf, p.45")
  * Pour résultats web: titre et URL (ex: "Source: [titre] - URL")

CONTEXTE FOURNI:
{context}

IMPORTANT: Le contexte ci-dessus contient TOUTES les informations nécessaires pour répondre.
Si des URLs sont présentes, cela signifie que tu AS ACCÈS à ces informations actualisées.
Réponds en utilisant UNIQUEMENT ces informations fournies, sans mentionner de limitations temporelles.
"""

    def generate(
        self,
        question: str,
        context: str,
        student_level: str = "Détaillé",
        retrieved_results: Optional[List[RetrievalResult]] = None,
        rigor_level: int = 3,
        num_examples: int = 2,
        include_proofs: bool = True,
        include_history: bool = False,
        detailed_latex: bool = True
    ) -> GeneratedResponse:
        """
        Génère une réponse pédagogique.

        Args:
            question: Question de l'utilisateur
            context: Contexte récupéré (from retriever)
            student_level: Niveau de détail (Simple/Détaillé/Beaucoup de détails)
            retrieved_results: Résultats bruts du retriever (pour métadonnées)

        Returns:
            GeneratedResponse avec réponse et sources

        Example:
            >>> response = generator.generate(
            ...     question="Qu'est-ce qu'une dérivée ?",
            ...     context=retriever_context,
            ...     student_level="L2"
            ... )
            >>> print(f"Réponse: {response.answer}")
            >>> print(f"Sources: {response.sources_cited}")
        """
        if not question or not question.strip():
            raise AgentError("Question vide fournie au generator", agent_name="generator")

        if not context or not context.strip():
            logger.warning("Contexte vide, réponse sera limitée")

        logger.info(
            f"Generating response",
            extra={
                "question_length": len(question),
                "context_length": len(context),
                "student_level": student_level
            }
        )

        try:
            # 1. Construire instructions personnalisées
            rigor_descriptions = {
                1: "Explications très intuitives et accessibles, privilégier l'intuition sur la rigueur",
                2: "Explications claires avec quelques éléments de rigueur",
                3: "Équilibre entre intuition et rigueur mathématique",
                4: "Explications rigoureuses avec justifications formelles",
                5: "Maximum de rigueur formelle, démonstrations complètes"
            }

            rigor_instruction = rigor_descriptions.get(rigor_level, rigor_descriptions[3])

            # Instructions dynamiques
            additional_instructions = []

            if num_examples > 0:
                additional_instructions.append(f"- Inclure EXACTEMENT {num_examples} exemple(s) concret(s) et détaillé(s)")
            else:
                additional_instructions.append("- Ne pas inclure d'exemples, rester théorique")

            if include_proofs:
                additional_instructions.append("- Inclure les démonstrations mathématiques détaillées")
            else:
                additional_instructions.append("- Ne pas inclure de démonstrations complètes, seulement les résultats")

            if include_history:
                additional_instructions.append("- Ajouter le contexte historique du concept (origine, découvreur, évolution)")

            if detailed_latex:
                additional_instructions.append("- Développer les formules LaTeX avec toutes les étapes intermédiaires")
            else:
                additional_instructions.append("- Utiliser des formules LaTeX concises")

            additional_instructions_text = "\n".join(additional_instructions)

            # 2. Construire prompt système enrichi
            system_prompt = self.system_prompt_template.format(
                level=student_level,
                context=context
            )

            # Ajouter instructions de personnalisation
            system_prompt += f"""

PARAMÈTRES DE PERSONNALISATION AVANCÉS:

RIGUEUR MATHÉMATIQUE (niveau {rigor_level}/5):
{rigor_instruction}

INSTRUCTIONS SPÉCIFIQUES:
{additional_instructions_text}

IMPORTANT: Respecte EXACTEMENT ces paramètres dans ta réponse.
"""

            # 3. Prompt utilisateur
            user_prompt = f"""QUESTION: {question}

Rappel: Réponds de manière pédagogique en citant tes sources et en préservant les formules LaTeX."""

            # 3. Générer avec LLM
            with log_performance(logger, "llm_generation"):
                llm_response = self.llm_client.generate(
                    prompt=user_prompt,
                    system=system_prompt
                )

            # 4. Post-traiter réponse
            full_answer = self._post_process_answer(llm_response.content)

            # 5. Générer suggestions séparément (appel LLM dédié)
            suggestions = self._generate_suggestions(question, full_answer)

            # 6. Extraire sources citées
            sources_cited = self._extract_cited_sources(
                full_answer,
                retrieved_results
            )

            # 7. Métadonnées
            metadata = {
                "student_level": student_level,
                "context_length": len(context),
                "has_formulas": self._detect_formulas(full_answer),
                "model_used": llm_response.model,
                "suggestions": suggestions  # Ajouter suggestions dans métadonnées
            }

            response = GeneratedResponse(
                answer=full_answer,
                sources_cited=sources_cited,
                llm_response=llm_response,
                metadata=metadata
            )

            logger.info(
                f"✓ Response generated",
                extra={
                    "answer_length": len(full_answer),
                    "sources_count": len(sources_cited),
                    "cost": llm_response.cost,
                    "tokens": llm_response.total_tokens
                }
            )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise AgentError(
                f"Response generation failed: {e}",
                agent_name="generator"
            ) from e

    def _post_process_answer(self, answer: str) -> str:
        """
        Post-traite la réponse générée.

        Opérations:
        - Nettoyer espaces multiples
        - Vérifier formules LaTeX valides
        - Formater citations

        Args:
            answer: Réponse brute du LLM

        Returns:
            Réponse nettoyée
        """
        # Nettoyer espaces
        answer = answer.strip()

        # TODO: Ajouter validation/correction LaTeX si nécessaire

        return answer

    def _extract_cited_sources(
        self,
        answer: str,
        retrieved_results: Optional[List[RetrievalResult]] = None,
        min_score: float = 0.4
    ) -> List[str]:
        """
        Extrait les sources réellement utilisées depuis les documents récupérés.

        IMPORTANT:
        - Utilise TOUJOURS les vrais documents récupérés (retrieved_results)
        - Filtre les documents avec un score de similarité trop faible
        - Si aucun document n'atteint le seuil minimum, retourne une liste vide

        Args:
            answer: Réponse générée (non utilisé, gardé pour compatibilité)
            retrieved_results: Résultats du retriever (SOURCE DE VÉRITÉ)
            min_score: Score minimum de similarité (0-1) pour considérer un document comme pertinent
                      Par défaut 0.4 (40% de similarité minimum)

        Returns:
            Liste de sources pertinentes (score >= min_score)

        Example:
            >>> sources = _extract_cited_sources(
            ...     answer="...",
            ...     retrieved_results=[result1, result2],
            ...     min_score=0.4
            ... )
            >>> print(sources)
            ['Analyse_L2.pdf (p.45)', 'Calcul.pdf (p.12)']
        """
        sources = []

        # Utiliser UNIQUEMENT les documents réellement récupérés ET pertinents
        if retrieved_results:
            for result in retrieved_results:
                # Filtrer par score de pertinence
                if result.score < min_score:
                    continue  # Skip les documents pas assez pertinents

                source = result.metadata.get("source", "Document inconnu")
                page = result.metadata.get("page")

                if page:
                    sources.append(f"{source} (p.{page})")
                else:
                    sources.append(source)

        # Dédupliquer tout en préservant l'ordre
        sources = list(dict.fromkeys(sources))

        return sources

    def _detect_formulas(self, text: str) -> bool:
        """
        Détecte si le texte contient des formules LaTeX.

        Args:
            text: Texte à analyser

        Returns:
            True si formules détectées
        """
        import re

        # Chercher délimiteurs LaTeX
        latex_patterns = [
            r'\$.*?\$',           # $...$
            r'\$\$.*?\$\$',       # $$...$$
            r'\\\[.*?\\\]',       # \[...\]
            r'\\\(.*?\\\)',       # \(...\)
            r'\\begin\{.*?\}',    # \begin{...}
        ]

        for pattern in latex_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True

        return False

    def _generate_suggestions(self, question: str, answer: str) -> List[str]:
        """
        Génère 3 suggestions de questions de suivi avec un appel LLM dédié.

        Args:
            question: Question originale de l'utilisateur
            answer: Réponse générée

        Returns:
            Liste de 3 suggestions de questions

        Example:
            >>> suggestions = self._generate_suggestions(
            ...     "Qu'est-ce qu'une dérivée ?",
            ...     "Une dérivée mesure..."
            ... )
            >>> print(suggestions)
            ['Comment calculer la dérivée de x² ?', ...]
        """
        import re

        # Prompt simple et direct pour générer uniquement des suggestions
        suggestions_prompt = f"""Génère exactement 3 questions de suivi pédagogiques basées sur cette conversation:

QUESTION ORIGINALE:
{question}

RÉPONSE FOURNIE (premiers 500 caractères):
{answer[:500]}...

CONSIGNES:
1. Génère exactement 3 questions pour approfondir le sujet
2. Les questions doivent être progressives: simple → intermédiaire → avancée
3. Format: une question par ligne, numérotée 1., 2., 3.
4. Questions courtes et claires
5. PAS de préambule ni explication, UNIQUEMENT les 3 questions

EXEMPLE DE FORMAT ATTENDU:
1. Comment calculer la dérivée de x² ?
2. Quelle est la relation entre dérivée et pente ?
3. Comment généraliser aux dérivées partielles ?

TES 3 QUESTIONS:"""

        try:
            # Appel LLM avec prompt minimal
            llm_response = self.llm_client.generate(
                prompt=suggestions_prompt,
                system="Tu es un assistant qui génère des questions pédagogiques. Réponds UNIQUEMENT avec 3 questions numérotées, rien d'autre."
            )

            # Parser les suggestions
            suggestions = []
            for line in llm_response.content.split('\n'):
                line = line.strip()
                # Chercher lignes format "1. Question"
                match = re.match(r'^\s*\d+\.\s*(.+)$', line)
                if match:
                    suggestions.append(match.group(1).strip())

            # Limiter à 3 suggestions
            return suggestions[:3] if len(suggestions) >= 3 else suggestions

        except Exception as e:
            logger.warning(f"Erreur lors de la génération des suggestions: {e}")
            # Suggestions par défaut en cas d'erreur
            return [
                "Pouvez-vous donner un exemple concret ?",
                "Comment ce concept s'applique-t-il dans d'autres contextes ?",
                "Quels sont les liens avec d'autres concepts mathématiques ?"
            ]

    def _extract_suggestions(self, answer: str) -> tuple[str, list[str]]:
        """
        Extrait les suggestions de questions de suivi de la réponse.

        Supporte 2 formats:
        1. Format avec marqueurs: ---SUGGESTIONS--- ... ---FIN-SUGGESTIONS---
        2. Format markdown: ### Suggestions de questions de suivi

        Args:
            answer: Réponse complète avec suggestions

        Returns:
            Tuple (réponse nettoyée, liste de suggestions)

        Example:
            >>> answer, suggestions = _extract_suggestions(full_answer)
            >>> print(suggestions)
            ['Question 1', 'Question 2', 'Question 3']
        """
        import re

        suggestions = []
        clean_answer = answer

        # FORMAT 1: Chercher le bloc avec marqueurs ---SUGGESTIONS---
        pattern_markers = r'---SUGGESTIONS---\s*(.*?)\s*---FIN-SUGGESTIONS---'
        match_markers = re.search(pattern_markers, answer, re.DOTALL)

        if match_markers:
            # Extraire les suggestions
            suggestions_block = match_markers.group(1)
            # Nettoyer la réponse (enlever le bloc de suggestions)
            clean_answer = re.sub(pattern_markers, '', answer, flags=re.DOTALL).strip()
        else:
            # FORMAT 2: Chercher le titre markdown "###Suggestions" ou "### Suggestions"
            # Pattern qui capture le titre et les lignes numérotées qui suivent
            pattern_markdown = r'###\s*Suggestions[^\n]*\n+((?:\d+\..*?\n?)+)'
            match_markdown = re.search(pattern_markdown, answer, re.DOTALL | re.IGNORECASE)

            if match_markdown:
                suggestions_block = match_markdown.group(1)  # Les suggestions elles-mêmes
                # Nettoyer la réponse (enlever le bloc de suggestions)
                clean_answer = re.sub(r'###\s*Suggestions.*$', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
            else:
                # Pas de suggestions trouvées
                return answer, []

        # Parser les suggestions (format: "1. Question")
        suggestion_pattern = r'^\s*\d+\.\s*(.+)$'

        for line in suggestions_block.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):  # Ignorer les titres markdown
                match = re.match(suggestion_pattern, line)
                if match:
                    suggestions.append(match.group(1).strip())

        return clean_answer, suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROMPT ENGINEERING:
# - Crucial pour qualité des réponses
# - Template avec placeholders {level}, {context}
# - Instructions claires et structurées
# - Exemples dans prompt si nécessaire
#
# RÈGLES IMPORTANTES DANS PROMPT:
# 1. Utiliser UNIQUEMENT contexte fourni (éviter hallucinations)
# 2. Citer sources (traçabilité)
# 3. Adapter niveau étudiant (pédagogie)
# 4. Préserver LaTeX exactement (important!)
# 5. Dire si info manquante (honnêteté)
#
# ADAPTATION NIVEAU:
# - Simple: Explications accessibles, vocabulaire simple, exemples concrets
# - Détaillé: Explications rigoureuses avec justifications, exemples variés
# - Beaucoup de détails: Explications complètes et formelles, démonstrations détaillées
#
# FORMULES LATEX:
# - Ne PAS modifier les formules du contexte
# - LLM peut parfois reformuler/corriger → risque d'erreurs
# - Prompt explicite: "préserve EXACTEMENT"
#
# SOURCES:
# - Format standardisé: [Source: fichier.pdf, page X]
# - Facilite extraction automatique
# - Permet vérification par l'utilisateur
# - Important pour confiance/traçabilité
#
# POST-PROCESSING:
# - Nettoyer espaces/formatage
# - Valider formules LaTeX (optionnel)
# - Extraire métadonnées (sources, formules, etc.)
#
# COÛT:
# - Dépend du modèle et longueur contexte
# - GPT-4o-mini: ~$0.0002-0.001 par question
# - GPT-4o: ~$0.005-0.015 par question
# - Claude Sonnet: ~$0.003-0.010 par question
#
# PERFORMANCE:
# - Latence: 1-3s selon modèle et tokens
# - GPT-4o-mini plus rapide (~1s)
# - Claude/GPT-4o plus lent mais meilleur (~2s)
#
# TEMPÉRATURE:
# - 0.3 (défaut): Reproductible, factuel
# - 0.0: Déterministe (même input → même output)
# - 0.7: Plus créatif (pour explications variées)
#
# EXTENSIONS:
# - Few-shot learning (exemples dans prompt)
# - Chain-of-thought (raisonnement étape par étape)
# - Self-consistency (générer plusieurs fois, voter)
# - Feedback loop (améliorer avec retours utilisateurs)
#
# DEBUGGING:
# ```python
# # Tester
# generator = GeneratorAgent(config, llm_client)
# response = generator.generate(
#     question="Qu'est-ce qu'une dérivée ?",
#     context="La dérivée mesure...",
#     student_level="L2"
# )
# print(f"Réponse: {response.answer}")
# print(f"Sources: {response.sources_cited}")
# print(f"Coût: ${response.llm_response.cost:.4f}")
# ```
#
# ═══════════════════════════════════════════════════════════════════════════════
