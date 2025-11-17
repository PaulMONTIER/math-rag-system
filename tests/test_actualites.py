#!/usr/bin/env python3
"""
Test question d'actualité mathématique 2024
"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("="*80)
print("TEST ACTUALITÉS MATHÉMATIQUES 2024")
print("="*80)

# Créer le workflow
print("\n[1/2] Création du workflow...")
config = load_config()
app = create_rag_workflow(config)
print("✓ Workflow créé\n")

# Test question actualité
print("[2/2] Test question actualité...")
question = "Actualités mathématiques 2024"
print(f"Question: {question}\n")

result = invoke_workflow(app, question, student_level="L2", rigor_level=3)

print(f"\n--- RÉSULTAT ---")
print(f"✅ Succès: {result['success']}")
print(f"Intent classifié: {result.get('intent', 'N/A')}")
print(f"Stratégie choisie: {result.get('search_strategy', 'N/A')}")
print(f"Confiance stratégie: {result.get('metadata', {}).get('planning', {}).get('confidence', 'N/A')}")

# Vérifier si web search a été utilisé
if result.get('web_search_results'):
    print(f"✓ Web search utilisé: {len(result['web_search_results'])} résultats")
else:
    print("❌ Web search PAS utilisé")

# Vérifier si retrieval local a été utilisé
retrieval_conf = result.get('metadata', {}).get('retrieval', {}).get('confidence')
if retrieval_conf is not None:
    print(f"⚠️  Retrieval local utilisé (confiance: {retrieval_conf:.2f})")
else:
    print("✓ Retrieval local PAS utilisé (correct pour WEB_ONLY)")

if result.get('final_response'):
    print(f"\nRéponse (300 premiers caractères):")
    print(f"{result['final_response'][:300]}...")

print("\n" + "="*80)

# Résumé du comportement attendu vs réel
print("\n📊 COMPORTEMENT ATTENDU vs RÉEL")
print("="*80)
print("ATTENDU:")
print("  - Intent: CURRENT_EVENT")
print("  - Stratégie: web_only")
print("  - Retrieval local: Non utilisé")
print("  - Web search: Utilisé")
print()
print("RÉEL:")
print(f"  - Intent: {result.get('intent', 'N/A')}")
print(f"  - Stratégie: {result.get('search_strategy', 'N/A')}")
print(f"  - Retrieval local: {'Utilisé' if retrieval_conf is not None else 'Non utilisé'}")
print(f"  - Web search: {'Utilisé' if result.get('web_search_results') else 'Non utilisé'}")
print()

# Verdict
intent_ok = result.get('intent') == 'CURRENT_EVENT'
strategy_ok = result.get('search_strategy') == 'web_only'
web_ok = result.get('web_search_results') is not None

if intent_ok and strategy_ok and web_ok:
    print("✅ SUCCÈS: Question d'actualité traitée correctement!")
else:
    print("⚠️  ATTENTION: Comportement inattendu")
    if not intent_ok:
        print(f"  ❌ Intent incorrect: {result.get('intent')} (attendu: CURRENT_EVENT)")
    if not strategy_ok:
        print(f"  ❌ Stratégie incorrecte: {result.get('search_strategy')} (attendu: web_only)")
    if not web_ok:
        print(f"  ❌ Web search non utilisé")

print("\n" + "="*80)
