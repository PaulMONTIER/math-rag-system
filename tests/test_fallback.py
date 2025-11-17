#!/usr/bin/env python3
"""
Test du système de fallback intelligent.

Ce test vérifie que le système déclenche automatiquement
une recherche web lorsque la confiance de retrieval est faible.
"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("="*80)
print("TEST FALLBACK INTELLIGENT")
print("="*80)

# Créer le workflow
print("\n[1/3] Création du workflow...")
config = load_config()
app = create_rag_workflow(config)
print("✓ Workflow créé\n")

# ============================================================================
# TEST 1: Question classique (confiance haute - PAS de fallback attendu)
# ============================================================================
print("="*80)
print("TEST 1: Question classique (haute confiance)")
print("="*80)
question1 = "Qu'est-ce qu'une dérivée?"
print(f"Question: {question1}\n")

result1 = invoke_workflow(app, question1, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT TEST 1 ---")
print(f"✅ Succès: {result1['success']}")
print(f"Intent: {result1.get('intent', 'N/A')}")
print(f"Stratégie: {result1.get('search_strategy', 'N/A')}")
print(f"Confiance retrieval: {result1.get('metadata', {}).get('retrieval', {}).get('confidence', 'N/A')}")
print(f"Fallback déclenché: {result1.get('metadata', {}).get('fallback', {}).get('triggered', False)}")
if result1.get('final_response'):
    print(f"\nRéponse (200 premiers caractères):")
    print(f"{result1['final_response'][:200]}...")

print("\n" + "="*80)

# ============================================================================
# TEST 2: Question hors-sujet (confiance faible - Fallback ATTENDU)
# ============================================================================
print("\nTEST 2: Question spécifique/rare (faible confiance)")
print("="*80)
question2 = "Qu'est-ce que la théorie des catégories supérieures en homotopie?"
print(f"Question: {question2}\n")

result2 = invoke_workflow(app, question2, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT TEST 2 ---")
print(f"✅ Succès: {result2['success']}")
print(f"Intent: {result2.get('intent', 'N/A')}")
print(f"Stratégie initiale: {result2.get('search_strategy', 'N/A')}")
print(f"Confiance retrieval: {result2.get('metadata', {}).get('retrieval', {}).get('confidence', 'N/A')}")
print(f"⭐ Fallback déclenché: {result2.get('metadata', {}).get('fallback', {}).get('triggered', False)}")
if result2.get('metadata', {}).get('fallback', {}).get('triggered'):
    print(f"   Web résultats ajoutés: {result2.get('metadata', {}).get('fallback', {}).get('web_results_added', 0)}")
if result2.get('final_response'):
    print(f"\nRéponse (200 premiers caractères):")
    print(f"{result2['final_response'][:200]}...")

print("\n" + "="*80)

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n📊 RÉSUMÉ DES TESTS")
print("="*80)

conf1 = result1.get('metadata', {}).get('retrieval', {}).get('confidence', 0)
conf2 = result2.get('metadata', {}).get('retrieval', {}).get('confidence', 0)
fallback1 = result1.get('metadata', {}).get('fallback', {}).get('triggered', False)
fallback2 = result2.get('metadata', {}).get('fallback', {}).get('triggered', False)

print(f"\nTest 1 (Dérivée):")
print(f"  Confiance: {conf1:.2f}")
print(f"  Fallback: {'❌ Non' if not fallback1 else '✅ Oui'} (attendu: Non)")

print(f"\nTest 2 (Catégories supérieures):")
print(f"  Confiance: {conf2:.2f}")
print(f"  Fallback: {'✅ Oui' if fallback2 else '❌ Non'} (attendu: Oui)")

print(f"\n{'✅ SUCCÈS' if conf1 > 0.6 and conf2 < 0.6 else '⚠️  ATTENTION'}: Le fallback intelligent est ")
print(f"{'fonctionnel' if conf1 > 0.6 and conf2 < 0.6 else 'à vérifier'}!")

print("\n" + "="*80)
