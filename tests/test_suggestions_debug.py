"""Test debug: Vérifier génération complète des suggestions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config
import json

# Charger config
config = load_config()

# Créer workflow
print("Création du workflow...")
workflow = create_rag_workflow(config)

# Question de test
question = "Qu'est-ce qu'une intégrale ?"

print(f"\n{'='*80}")
print(f"QUESTION: {question}")
print(f"{'='*80}\n")

# Invoquer workflow
result = invoke_workflow(
    workflow=workflow,
    question=question,
    student_level="Détaillé"
)

if result["success"]:
    print("✓ Workflow exécuté avec succès\n")

    # Afficher structure complète des métadonnées
    print(f"{'='*80}")
    print("STRUCTURE DES MÉTADONNÉES:")
    print(f"{'='*80}")
    print(json.dumps(result.get("metadata", {}), indent=2, ensure_ascii=False))

    # Vérifier où sont les suggestions
    print(f"\n{'='*80}")
    print("LOCALISATION DES SUGGESTIONS:")
    print(f"{'='*80}")

    # Dans metadata.generation.suggestions
    gen_meta = result.get("metadata", {}).get("generation", {})
    suggestions = gen_meta.get("suggestions", [])

    if suggestions:
        print(f"✓ TROUVÉ {len(suggestions)} suggestions dans metadata.generation.suggestions:")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")
    else:
        print("❌ AUCUNE suggestion dans metadata.generation.suggestions")
        print(f"   Contenu de generation: {gen_meta}")

else:
    print(f"❌ Erreur: {result.get('error')}")
