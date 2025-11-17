#!/usr/bin/env python3
"""
Test avec une vraie question WEB qui devrait être classée comme CURRENT_EVENT
"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("="*80)
print("TEST WEB - Question événement actuel")
print("="*80)

# Créer le workflow
print("\n[1/2] Création du workflow...")
config = load_config()
app = create_rag_workflow(config)
print("✓ Workflow créé")

# Test avec une question qui DEVRAIT être classée comme CURRENT_EVENT
print("\n[2/2] Test question web...")
question = "Quelles sont les dernières avancées récentes en théorie des nombres?"
print(f"Question: {question}")

result = invoke_workflow(app, question, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT ---")
print(f"Succès: {result['success']}")
print(f"Intent: {result.get('intent', 'N/A')}")
print(f"Stratégie: {result.get('search_strategy', 'N/A')}")
print(f"Confiance: {result.get('confidence_score', 'N/A')}")
if result.get('final_response'):
    print(f"\nRéponse (300 premiers caractères):")
    print(f"{result['final_response'][:300]}...")
if result.get('error_message'):
    print(f"❌ Erreur: {result['error_message']}")

print("\n" + "="*80)
