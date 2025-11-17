"""Debug: Vérifier les sources retournées par le système."""
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
    print("SOURCES RETOURNÉES:")
    print(f"{'='*80}")

    sources = result.get("sources_cited", [])

    if sources:
        print(f"✓ {len(sources)} source(s) trouvée(s):")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source}")
    else:
        print("❌ AUCUNE source retournée")

    print(f"\n{'='*80}")
    print("MÉTADONNÉES DE RETRIEVAL:")
    print(f"{'='*80}")

    retrieval_meta = result.get("metadata", {}).get("retrieval", {})
    print(f"Documents trouvés: {retrieval_meta.get('docs_found', 0)}")
    print(f"Score moyen: {retrieval_meta.get('avg_score', 0):.3f}")

    # Vérifier si on a accès aux documents récupérés dans le state
    print(f"\n{'='*80}")
    print("STRUCTURE COMPLÈTE DU RÉSULTAT:")
    print(f"{'='*80}")
    print("Clés disponibles:", list(result.keys()))

    # Afficher toutes les métadonnées
    print(f"\n{'='*80}")
    print("TOUTES LES MÉTADONNÉES:")
    print(f"{'='*80}")
    print(json.dumps(result.get("metadata", {}), indent=2, ensure_ascii=False))

else:
    print(f"❌ Erreur: {result.get('error')}")
