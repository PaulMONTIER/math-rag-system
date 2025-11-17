#!/usr/bin/env python3
"""
Test pour vérifier que le traçage manuel Langfuse fonctionne correctement.

Ce test exécute une vraie requête à travers le workflow et vérifie:
- La trace principale est créée
- Les spans de workflow sont créés (classify, plan, retrieve, generate, etc.)
- Les generations OpenAI sont imbriquées dans les spans
- Tout est correctement finalisé avec outputs et usage
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.utils.langfuse_integration import get_langfuse_client, is_langfuse_enabled
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config

print("=" * 80)
print("TEST TRAÇAGE MANUEL LANGFUSE - WORKFLOW COMPLET")
print("=" * 80)

# Vérifier que Langfuse est activé
if not is_langfuse_enabled():
    print("❌ Langfuse non activé - configurez les clés API dans .env")
    print("\nVariables requises:")
    print("  LANGFUSE_PUBLIC_KEY")
    print("  LANGFUSE_SECRET_KEY")
    print("  LANGFUSE_BASE_URL")
    sys.exit(1)

client = get_langfuse_client()
if not client:
    print("❌ Impossible d'initialiser le client Langfuse")
    sys.exit(1)

print("\n✓ Client Langfuse initialisé")

# Charger la config
try:
    config = load_config()
    print("✓ Configuration chargée")
except Exception as e:
    print(f"❌ Erreur lors du chargement de la config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Créer le workflow
try:
    workflow = create_rag_workflow(config)
    print("✓ Workflow MathRAG initialisé")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du workflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Question de test simple
test_question = "Qu'est-ce qu'un espace vectoriel?"

print(f"\n📝 Question de test: '{test_question}'")
print("\n🔄 Exécution du workflow...")

try:
    # Exécuter le workflow
    result = invoke_workflow(workflow, test_question)

    print("\n✅ Workflow exécuté avec succès!")
    print("\n" + "=" * 80)
    print("RÉSULTAT:")
    print("=" * 80)

    # Afficher le résultat
    if result.get("generated_answer"):
        print(f"\n📄 Réponse générée ({len(result['generated_answer'])} caractères):")
        print(result["generated_answer"][:200] + "...")

    if result.get("sources_cited"):
        print(f"\n📚 Sources citées: {len(result['sources_cited'])}")

    if result.get("confidence_score"):
        print(f"\n🎯 Score de confiance: {result['confidence_score']:.2%}")

    # Vérifier les informations Langfuse
    if result.get("langfuse_trace_id"):
        print(f"\n🔍 Trace Langfuse ID: {result['langfuse_trace_id']}")
        print(f"🔍 Trace URL: {os.getenv('LANGFUSE_BASE_URL')}/trace/{result['langfuse_trace_id']}")

    print("\n" + "=" * 80)
    print("VÉRIFICATION LANGFUSE")
    print("=" * 80)

    # Flush pour s'assurer que tout est envoyé
    client.flush()
    print("\n✓ Données envoyées à Langfuse")

    print("\n📊 Dans votre dashboard Langfuse, vous devriez voir:")
    print("  math_rag_query (trace)")
    print("  ├─ classify (span)")
    print("  │  └─ openai_call (generation) ← IMBRIQUÉ!")
    print("  ├─ plan (span)")
    print("  │  └─ openai_call (generation) ← IMBRIQUÉ!")
    print("  ├─ retrieve (span)")
    print("  ├─ generate (span)")
    print("  │  └─ openai_call (generation) ← IMBRIQUÉ!")
    print("  ├─ editor (span)")
    print("  │  └─ openai_call (generation) ← IMBRIQUÉ!")
    print("  └─ verify (span)")
    print("     └─ openai_call (generation) ← IMBRIQUÉ!")

    print(f"\n🌐 Allez sur: {os.getenv('LANGFUSE_BASE_URL')}")
    if result.get("langfuse_trace_id"):
        print(f"🔗 Lien direct: {os.getenv('LANGFUSE_BASE_URL')}/trace/{result['langfuse_trace_id']}")
    else:
        print("Cherchez la trace 'math_rag_query' la plus récente")

    print("\n" + "=" * 80)
    print("✅ TEST TERMINÉ AVEC SUCCÈS")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Erreur lors de l'exécution du workflow: {e}")
    import traceback
    traceback.print_exc()

    # Essayer quand même de flush
    try:
        client.flush()
    except:
        pass

    sys.exit(1)
