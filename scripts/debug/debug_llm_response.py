"""Debug: Afficher la réponse brute complète du LLM."""
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

print(f"\n{'='*80}")
print(f"QUESTION: {question}")
print(f"{'='*80}\n")

# Invoquer workflow
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
    # Afficher TOUTE la réponse
    full_response = result["final_response"]

    print(f"RÉPONSE COMPLÈTE DU LLM:")
    print(f"{'='*80}")
    print(full_response)
    print(f"{'='*80}\n")

    # Vérifier la présence de marqueurs
    print(f"ANALYSE:")
    print(f"  Longueur totale: {len(full_response)} caractères")
    print(f"  Contient '---SUGGESTIONS---': {'OUI' if '---SUGGESTIONS---' in full_response else 'NON'}")
    print(f"  Contient '---FIN-SUGGESTIONS---': {'OUI' if '---FIN-SUGGESTIONS---' in full_response else 'NON'}")

    # Afficher les 500 derniers caractères
    print(f"\n{'='*80}")
    print("DERNIERS 500 CARACTÈRES DE LA RÉPONSE:")
    print(f"{'='*80}")
    print(full_response[-500:])
    print(f"{'='*80}\n")
else:
    print(f"❌ Erreur: {result.get('error')}")
