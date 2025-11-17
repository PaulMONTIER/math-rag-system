"""Test: Vérifier que le système de suggestions fonctionne."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config

# Charger config
config = load_config()

# Créer workflow
print("Création du workflow...")
workflow = create_rag_workflow(config)

# Question de test simple
question = "Qu'est-ce qu'une dérivée ?"

print(f"\n{'='*80}")
print(f"TEST DES SUGGESTIONS")
print(f"{'='*80}\n")
print(f"Question: {question}\n")

# Invoquer workflow avec paramètres par défaut
result = invoke_workflow(
    workflow=workflow,
    question=question,
    student_level="Détaillé",
    rigor_level=3,
    num_examples=2,
    include_proofs=True,
    include_history=False,
    detailed_latex=True
)

if result["success"]:
    print("✓ Workflow exécuté avec succès\n")

    # Vérifier la réponse
    answer = result["final_response"]
    print(f"RÉPONSE (premiers 300 caractères):")
    print(answer[:300])
    print(f"... (longueur totale: {len(answer)} caractères)\n")

    # Vérifier les métadonnées
    metadata = result.get("metadata", {})
    print(f"MÉTADONNÉES DISPONIBLES:")
    for key in metadata.keys():
        print(f"  - {key}")

    # Vérifier spécifiquement les suggestions
    print(f"\n{'='*80}")
    print("VÉRIFICATION DES SUGGESTIONS:")
    print(f"{'='*80}\n")

    if "generation" in metadata:
        gen_metadata = metadata["generation"]
        suggestions = gen_metadata.get("suggestions", [])

        print(f"Suggestions trouvées: {len(suggestions)}")

        if suggestions:
            print("\n✓ SUGGESTIONS GÉNÉRÉES:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("\n❌ Aucune suggestion trouvée dans les métadonnées")

            # Vérifier si suggestions sont dans la réponse brute
            if "---SUGGESTIONS---" in answer:
                print("\n⚠️  Le marqueur de suggestions est PRÉSENT dans la réponse")
                print("    → Le problème vient de l'extraction")
            else:
                print("\n⚠️  Le marqueur de suggestions N'EST PAS dans la réponse")
                print("    → Le LLM n'a pas généré de suggestions")
    else:
        print("❌ Pas de métadonnées 'generation' trouvées")

    # Afficher toutes les métadonnées pour debug
    print(f"\n{'='*80}")
    print("DUMP COMPLET DES MÉTADONNÉES:")
    print(f"{'='*80}\n")
    import json
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

else:
    print(f"❌ Erreur: {result.get('error')}")

print(f"\n{'='*80}")
print("TEST TERMINÉ")
print(f"{'='*80}")
