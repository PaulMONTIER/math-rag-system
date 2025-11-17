#!/usr/bin/env python3
"""
Debug test pour comprendre l'erreur NoneType + NoneType
"""

import traceback
from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("="*80)
print("DEBUG TEST - Web Search Error")
print("="*80)

try:
    # 1. Créer le workflow
    print("\n[1/2] Création du workflow...")
    config = load_config()
    app = create_rag_workflow(config)
    print("✓ Workflow créé")

    # 2. Test question web qui échoue
    print("\n[2/2] Test question web...")
    question = "Actualités mathématiques 2024"
    print(f"Question: {question}")

    result = invoke_workflow(app, question, student_level="L2", rigor_level=3)

    print(f"\n--- RÉSULTAT ---")
    print(f"Succès: {result['success']}")
    print(f"Stratégie: {result.get('search_strategy', 'N/A')}")
    if result.get('final_response'):
        print(f"Réponse: {result['final_response'][:200]}...")
    if result.get('error_message'):
        print(f"❌ Erreur: {result['error_message']}")

except Exception as e:
    print("\n❌ EXCEPTION CAPTURÉE:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    print("\nTRACEBACK COMPLET:")
    traceback.print_exc()

print("\n" + "="*80)
