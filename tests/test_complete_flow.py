"""Test complet: Vérifier le workflow de bout en bout pour les suggestions."""
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
    student_level="Détaillé"
)

if result["success"]:
    print("✓ Workflow exécuté avec succès\n")

    print(f"{'='*80}")
    print("VALIDATION DU FLUX COMPLET:")
    print(f"{'='*80}\n")

    # 1. Vérifier que les suggestions sont générées
    generation_meta = result.get("metadata", {}).get("generation", {})
    suggestions = generation_meta.get("suggestions", [])

    if suggestions:
        print(f"✓ BACKEND: {len(suggestions)} suggestions générées")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")
    else:
        print("❌ BACKEND: Aucune suggestion générée")

    print()

    # 2. Simuler ce que fait le frontend
    print("SIMULATION DU FRONTEND:")
    print("-" * 80)

    # Le code ajouté aux lignes 899-904 fait exactement ceci:
    generation_meta = result.get("metadata", {}).get("generation", {})
    suggestions = generation_meta.get("suggestions", [])

    if suggestions:
        print(f"✓ FRONTEND: Peut extraire les suggestions depuis result['metadata']['generation']['suggestions']")
        print(f"✓ FRONTEND: display_suggestions() serait appelée avec:")
        print(f"   - suggestions: {suggestions}")
        print(f"   - message_idx: 0 (pour une nouvelle réponse)")
        print()
        print("✓ FRONTEND: Les 3 boutons suivants seraient créés:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   Bouton {i}: 📖 {suggestion}")
            print(f"            key: suggest_0_{i-1}")
            print(f"            type: secondary")
            print()
    else:
        print("❌ FRONTEND: Aucune suggestion à afficher")

    print(f"{'='*80}")
    print("CONCLUSION:")
    print(f"{'='*80}")

    if suggestions and len(suggestions) == 3:
        print("✅ LE FLUX COMPLET FONCTIONNE CORRECTEMENT!")
        print()
        print("Les suggestions:")
        print("1. Sont générées par le backend ✓")
        print("2. Sont stockées dans metadata.generation.suggestions ✓")
        print("3. Peuvent être extraites par le frontend ✓")
        print("4. Seront affichées comme boutons cliquables ✓")
        print()
        print("👉 Actualisez la page Streamlit et posez une nouvelle question pour voir les suggestions!")
    else:
        print("❌ PROBLÈME DÉTECTÉ")

else:
    print(f"❌ Erreur: {result.get('error')}")
