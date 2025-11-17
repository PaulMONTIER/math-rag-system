#!/usr/bin/env python3
"""
Test simple du workflow complet
"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("="*80)
print("TEST SIMPLE - Math RAG System (Project A)")
print("="*80)

# 1. Créer le workflow
print("\n[1/3] Création du workflow...")
config = load_config()
app = create_rag_workflow(config)
print("✓ Workflow créé avec 8 agents et routing intelligent")

# 2. Test question locale (RAG)
print("\n[2/3] Test question locale (RAG)...")
question1 = "Qu'est-ce qu'une dérivée?"
print(f"Question: {question1}")

result1 = invoke_workflow(app, question1, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT ---")
print(f"Stratégie utilisée: {result1.get('search_strategy', 'N/A')}")
print(f"Qualité Editor: {result1.get('editor_quality_score', 'N/A'):.2f}")
print(f"Confiance: {result1.get('confidence_score', 'N/A'):.2f}")
print(f"\nRéponse: {result1['final_response'][:200]}...")
print(f"Sources: {len(result1.get('sources_cited', []))} documents")

# 3. Test question web
print("\n[3/3] Test question web...")
question2 = "Quelles sont les actualités récentes en mathématiques en 2024?"
print(f"Question: {question2}")

result2 = invoke_workflow(app, question2, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT ---")
print(f"Stratégie utilisée: {result2.get('search_strategy', 'N/A')}")
print(f"Qualité Editor: {result2.get('editor_quality_score', 'N/A'):.2f}")
print(f"\nRéponse: {result2['final_response'][:200]}...")
print(f"Web results: {len(result2.get('web_search_results', []))} résultats")

print("\n" + "="*80)
print("✅ TEST TERMINÉ")
print("="*80)
