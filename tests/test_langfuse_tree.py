#!/usr/bin/env python3
"""
Test pour vérifier la structure hiérarchique complète dans Langfuse.

Ce script teste que nous voyons bien:
- La trace principale (math_rag_query)
- Les spans de workflow (classify, plan, retrieve, generate, etc.)
- Les generations OpenAI imbriquées dans les spans
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.utils.langfuse_integration import get_langfuse_client, is_langfuse_enabled

print("=" * 80)
print("TEST STRUCTURE HIÉRARCHIQUE LANGFUSE")
print("=" * 80)

if not is_langfuse_enabled():
    print("❌ Langfuse non activé - configurez les clés API dans .env")
    exit(1)

client = get_langfuse_client()
if not client:
    print("❌ Impossible d'initialiser le client Langfuse")
    exit(1)

print("\n✓ Client Langfuse initialisé")

# Créer une trace principale
trace = client.trace(
    name="test_hierarchical_structure",
    metadata={"test": "hierarchy"}
)

print(f"✓ Trace créée: {trace.id}")

# Créer un span de premier niveau
classify_span = trace.span(
    name="classify_step",
    input={"question": "Test question"}
)

print("✓ Span 'classify_step' créé")

# Créer une génération imbriquée dans le span
generation = classify_span.generation(
    name="classify_llm_call",
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Tu es un classificateur"},
        {"role": "user", "content": "Question: Test"}
    ]
)

print("✓ Generation 'classify_llm_call' créée sous le span")

# Finaliser la génération
generation.end(
    output="MATH_QUESTION",
    usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
)

print("✓ Generation finalisée")

# Finaliser le span
classify_span.end(
    output={"intent": "MATH_QUESTION", "confidence": 0.95}
)

print("✓ Span 'classify_step' finalisé")

# Créer un autre span
generate_span = trace.span(
    name="generate_step",
    input={"context": "Some context"}
)

print("✓ Span 'generate_step' créé")

# Génération imbriquée
gen2 = generate_span.generation(
    name="generate_llm_call",
    model="gpt-4o",
    input=[{"role": "user", "content": "Generate answer"}]
)

gen2.end(output="Generated response", usage={"total_tokens": 100})
generate_span.end(output={"answer": "Generated response"})

print("✓ Span 'generate_step' finalisé")

# Finaliser la trace
trace.update(status="SUCCESS")

print("\n✓ Trace finalisée avec succès")

# Flush pour s'assurer que tout est envoyé
client.flush()

print("\n" + "=" * 80)
print("✅ TEST TERMINÉ")
print("=" * 80)
print("\nDans votre dashboard Langfuse, vous devriez voir:")
print("  test_hierarchical_structure (trace)")
print("  ├─ classify_step (span)")
print("  │  └─ classify_llm_call (generation)")
print("  └─ generate_step (span)")
print("     └─ generate_llm_call (generation)")
print(f"\nAllez sur: {os.getenv('LANGFUSE_BASE_URL')}")
print("Cherchez la trace 'test_hierarchical_structure'")
print("\n" + "=" * 80)
