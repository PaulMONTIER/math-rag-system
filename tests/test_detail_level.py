"""Test: Vérifier que le niveau de détail fonctionne correctement."""
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

# Tester chaque niveau
levels = ["Simple", "Détaillé", "Beaucoup de détails"]

for level in levels:
    print(f"\n{'='*80}")
    print(f"TEST NIVEAU: {level}")
    print(f"{'='*80}\n")

    # Invoquer workflow avec ce niveau
    result = invoke_workflow(
        workflow=workflow,
        question=question,
        student_level=level
    )

    if result["success"]:
        answer = result["final_response"]
        print(f"RÉPONSE (premiers 500 caractères):")
        print(answer[:500])
        print(f"\n... (longueur totale: {len(answer)} caractères)")

        # Vérifier que le niveau est bien dans les métadonnées
        if "generation" in result.get("metadata", {}):
            print(f"\n✓ Niveau utilisé: {result['metadata'].get('student_level', 'NON TROUVÉ')}")

        print(f"✓ Tokens: {result['metadata'].get('generation', {}).get('tokens', 'N/A')}")
    else:
        print(f"❌ Erreur: {result.get('error')}")

print(f"\n{'='*80}")
print("TEST TERMINÉ")
print(f"{'='*80}")
