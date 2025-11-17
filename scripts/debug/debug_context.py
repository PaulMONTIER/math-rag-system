#!/usr/bin/env python3
"""
Debug script pour voir le contexte passé au generator
"""

from src.utils.config_loader import load_config
from src.agents.web_searcher import WebSearchAgent

# Charger config
config = load_config()

# Créer web searcher
web_searcher = WebSearchAgent(config)

# Test recherche
question = "Actualités mathématiques 2024"
print(f"Question: {question}\n")

# 1. Recherche classique
response = web_searcher.search(question, max_results=5)
print(f"=== Web Search Results ===")
print(f"Found {len(response.results)} results\n")

for idx, r in enumerate(response.results, 1):
    print(f"{idx}. {r.title}")
    print(f"   URL: {r.url}")
    print(f"   Snippet: {r.snippet[:100]}...")
    print()

# 2. Contexte formaté
print("=== search_for_context() output ===")
context = web_searcher.search_for_context(question)
print(f"Length: {len(context)} characters\n")
print(context)
print("\n" + "="*80)
