"""Test simple de la génération de suggestions."""
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

# Question de test
question = "Qu'est-ce qu'une dérivée ?"

print(f"\nQUESTION: {question}\n")

# Invoquer workflow
result = invoke_workflow(
    workflow=workflow,
    question=question,
    student_level="Détaillé"
)

if result["success"]:
    print("✓ Workflow exécuté avec succès\n")

    # Afficher toutes les clés du résultat
    print("CLÉS DU RÉSULTAT:")
    for key in result.keys():
        print(f"  - {key}")

    # Afficher métadonnées complètes
    print(f"\nMÉTADONNÉES:")
    import json
    print(json.dumps(result.get("metadata", {}), indent=2, ensure_ascii=False))

    # Chercher suggestions partout
    print(f"\n{'='*80}")
    print("RECHERCHE DE SUGGESTIONS:")
    print(f"{'='*80}")

    # Dans metadata root
    if "suggestions" in result.get("metadata", {}):
        print(f"✓ Trouvé dans metadata['suggestions']:")
        for i, s in enumerate(result["metadata"]["suggestions"], 1):
            print(f"   {i}. {s}")
    else:
        print(f"❌ PAS dans metadata['suggestions']")

    # Dans metadata.generation
    if "generation" in result.get("metadata", {}) and "suggestions" in result["metadata"]["generation"]:
        print(f"✓ Trouvé dans metadata['generation']['suggestions']:")
        for i, s in enumerate(result["metadata"]["generation"]["suggestions"], 1):
            print(f"   {i}. {s}")
    else:
        print(f"❌ PAS dans metadata['generation']['suggestions']")

else:
    print(f"❌ Erreur: {result.get('error')}")
