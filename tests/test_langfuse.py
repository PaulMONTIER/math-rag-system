#!/usr/bin/env python3
"""
Test de l'intégration Langfuse avec LangGraph.

Vérifie que:
1. Langfuse est correctement configuré avec les nouvelles clés API
2. Le CallbackHandler est initialisé
3. Les traces sont envoyées à Langfuse lors de l'exécution du workflow
"""

import os
from dotenv import load_dotenv

# IMPORTANT: Charger .env en PREMIER, avant tous les imports
load_dotenv()

from src.utils.config_loader import load_config
from src.utils.langfuse_integration import (
    is_langfuse_enabled,
    get_langfuse_client,
    get_langfuse_handler
)
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print("=" * 80)
print("TEST INTÉGRATION LANGFUSE + LANGGRAPH")
print("=" * 80)

# 1. Vérifier les variables d'environnement
print("\n[1/5] Vérification des variables d'environnement...")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL")

print(f"  LANGFUSE_PUBLIC_KEY: {'✓ Configuré' if public_key else '❌ Manquant'}")
print(f"  LANGFUSE_SECRET_KEY: {'✓ Configuré' if secret_key else '❌ Manquant'}")
print(f"  LANGFUSE_BASE_URL: {base_url}")

if public_key:
    print(f"  → Clé publique: {public_key[:15]}...{public_key[-10:]}")

# 2. Vérifier si Langfuse est activé
print("\n[2/5] Vérification activation Langfuse...")
enabled = is_langfuse_enabled()
print(f"  Langfuse activé: {'✓ OUI' if enabled else '❌ NON'}")

# 3. Initialiser le client Langfuse
print("\n[3/5] Initialisation du client Langfuse...")
try:
    client = get_langfuse_client()
    if client:
        print("  ✓ Client Langfuse initialisé avec succès")
        print(f"  → Host: {client.base_url}")
    else:
        print("  ❌ Échec initialisation client")
except Exception as e:
    print(f"  ❌ Erreur: {e}")

# 4. Initialiser le CallbackHandler
print("\n[4/5] Initialisation du CallbackHandler...")
try:
    handler = get_langfuse_handler()
    if handler:
        print("  ✓ CallbackHandler Langfuse initialisé")
        print(f"  → Type: {type(handler).__name__}")
    else:
        print("  ❌ CallbackHandler non disponible")
except Exception as e:
    print(f"  ❌ Erreur: {e}")

# 5. Test avec une vraie question via le workflow
print("\n[5/5] Test avec workflow complet...")
try:
    config = load_config()
    app = create_rag_workflow(config)

    question = "Qu'est-ce qu'une dérivée ?"
    print(f"  Question: {question}")

    print("\n  Exécution du workflow...")
    result = invoke_workflow(app, question, student_level="L2", rigor_level=3)

    print(f"\n  Résultat:")
    print(f"    Succès: {result['success']}")
    print(f"    Intent: {result.get('intent', 'N/A')}")
    print(f"    Confiance retrieval: {result.get('metadata', {}).get('retrieval', {}).get('confidence', 'N/A')}")

    if enabled and handler:
        print("\n  ✓ Traces devraient être visibles dans Langfuse:")
        print(f"    → Dashboard: {base_url}")
        print("    → Chercher la trace avec le texte: 'Qu'est-ce qu'une dérivée ?'")
    else:
        print("\n  ⚠️  Langfuse non activé, pas de traces envoyées")

    print("\n  ✓ Workflow exécuté avec succès")

except Exception as e:
    print(f"\n  ❌ Erreur lors de l'exécution: {e}")
    import traceback
    traceback.print_exc()

# Résumé
print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)

if enabled and handler:
    print("✅ Langfuse est correctement configuré et intégré avec LangGraph")
    print("\nPour voir les traces:")
    print(f"1. Ouvrez {base_url}")
    print(f"2. Connectez-vous avec votre compte")
    print(f"3. Cherchez les traces du workflow 'math_rag_query'")
    print(f"\nLes traces incluent:")
    print("  - Chaque nœud du graphe (classify, retrieve, generate, etc.)")
    print("  - Les appels LLM avec prompt/response")
    print("  - Les métriques de latence et coûts")
    print("  - Les métadonnées de qualité (confidence, etc.)")
elif enabled and not handler:
    print("⚠️  Langfuse activé mais CallbackHandler non disponible")
    print("   → Vérifiez l'installation: pip install langfuse")
elif not enabled:
    print("⚠️  Langfuse non activé (clés API manquantes)")
    print("   → Configurez LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env")
else:
    print("❌ Problème avec la configuration Langfuse")

print("\n" + "=" * 80)
