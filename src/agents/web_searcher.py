"""
Agent de recherche web - Recherche externe via DuckDuckGo.

Cet agent permet de rechercher des informations sur le web lorsque
la base vectorielle locale ne contient pas d'informations pertinentes.

Usage:
    from src.agents.web_searcher import WebSearchAgent

    agent = WebSearchAgent(config)
    results = agent.search("Qu'est-ce que le théorème de Fermat?")
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from duckduckgo_search import DDGS

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WebSearchResult:
    """Résultat d'une recherche web."""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0


@dataclass
class WebSearchResponse:
    """Réponse complète de la recherche web."""
    query: str
    results: List[WebSearchResult]
    total_results: int
    summary: str
    sources: List[Dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════════
# Web Search Agent
# ═══════════════════════════════════════════════════════════════════════════════

class WebSearchAgent:
    """
    Agent de recherche web utilisant DuckDuckGo.

    Caractéristiques:
    - Pas besoin de clé API
    - Recherche anonyme
    - Respect de la vie privée
    - Résultats pertinents et récents
    """

    def __init__(
        self,
        config: object,
        max_results: int = 5,
        timeout: int = 10
    ):
        """
        Initialise l'agent de recherche web.

        Args:
            config: Configuration du système
            max_results: Nombre maximum de résultats à retourner
            timeout: Timeout en secondes pour chaque recherche
        """
        self.config = config
        self.max_results = max_results
        self.timeout = timeout

        logger.info("WebSearchAgent initialized")

    def search(self, query: str, max_results: Optional[int] = None) -> WebSearchResponse:
        """
        Effectue une recherche web.

        Args:
            query: Question ou requête de recherche
            max_results: Nombre de résultats (override default)

        Returns:
            WebSearchResponse avec les résultats

        Example:
            >>> agent = WebSearchAgent(config)
            >>> response = agent.search("calculus derivative definition")
            >>> print(f"Found {response.total_results} results")
        """
        num_results = max_results or self.max_results
        logger.info(f"Searching web: '{query}' (max {num_results} results)")

        try:
            # Recherche DuckDuckGo
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    max_results=num_results,
                    region='wt-wt',  # Worldwide
                    safesearch='moderate',
                    timelimit=None  # Pas de limite temporelle
                ))

            # Convertir en WebSearchResult
            results = []
            for idx, result in enumerate(search_results):
                web_result = WebSearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('href', ''),
                    snippet=result.get('body', 'No snippet available'),
                    relevance_score=1.0 - (idx * 0.1)  # Score décroissant
                )
                results.append(web_result)

            # Créer le résumé
            summary = self._create_summary(query, results)

            # Créer les sources
            sources = [
                {
                    "title": r.title,
                    "url": r.url,
                    "type": "web"
                }
                for r in results
            ]

            logger.info(f"✓ Found {len(results)} web results")

            return WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                summary=summary,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            # Retourner résultat vide en cas d'erreur
            return WebSearchResponse(
                query=query,
                results=[],
                total_results=0,
                summary="Web search failed. Please try again.",
                sources=[]
            )

    def _create_summary(self, query: str, results: List[WebSearchResult]) -> str:
        """
        Crée un résumé textuel des résultats.

        Args:
            query: Requête originale
            results: Résultats de recherche

        Returns:
            Résumé formaté
        """
        if not results:
            return f"No web results found for: {query}"

        summary_parts = []
        summary_parts.append(f"Found {len(results)} web results for: {query}\n")

        for idx, result in enumerate(results[:3], 1):  # Top 3 results
            summary_parts.append(f"\n{idx}. **{result.title}**")
            summary_parts.append(f"   {result.snippet}")
            summary_parts.append(f"   Source: {result.url}")

        return "\n".join(summary_parts)

    def search_for_context(self, query: str) -> str:
        """
        Recherche web et retourne le contexte formaté pour le LLM.

        Args:
            query: Question de recherche

        Returns:
            Contexte textuel formaté

        Example:
            >>> context = agent.search_for_context("derivative calculus")
            >>> # Utilise context comme prompt pour le LLM
        """
        response = self.search(query)

        if not response.results:
            return "No relevant web results found."

        # Formatter pour le contexte LLM
        context_parts = []
        context_parts.append(f"# Web Search Results for: {query}\n")

        for idx, result in enumerate(response.results, 1):
            context_parts.append(f"## Result {idx}: {result.title}")
            context_parts.append(f"**URL**: {result.url}")
            context_parts.append(f"**Snippet**: {result.snippet}\n")

        return "\n".join(context_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# DUCKDUCKGO SEARCH:
# - Gratuit, pas de clé API
# - Respecte la vie privée
# - Limite: ~50-100 requêtes/minute
# - Résultats de qualité variable
#
# ALTERNATIVES:
# - Tavily: Meilleure qualité, nécessite clé API ($)
# - SerpAPI: Très complet, nécessite clé API ($$)
# - Brave Search: Bon compromis, clé API gratuite limitée
#
# AMÉLIRATIONS POSSIBLES:
# 1. Ajouter caching des résultats (Redis/SQLite)
# 2. Ranking/filtering par pertinence (BM25, embeddings)
# 3. Extraction de contenu complet (BeautifulSoup)
# 4. Support multi-langues (region parameter)
# 5. Retry logic avec backoff exponentiel
#
# INTÉGRATION WORKFLOW:
# - Utilisé quand RAG local insuffisant
# - Combinaison RAG local + Web search
# - Fallback en cas d'erreur
#
# ═══════════════════════════════════════════════════════════════════════════════
