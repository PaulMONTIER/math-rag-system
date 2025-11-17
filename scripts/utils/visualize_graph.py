#!/usr/bin/env python3
"""
Visualisation du graphe workflow
"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow

print("Création du workflow...")
config = load_config()
app = create_rag_workflow(config)

print("\n" + "="*80)
print("STRUCTURE DU GRAPHE WORKFLOW")
print("="*80)

# Récupérer le graphe
graph = app.get_graph()

# Afficher les nœuds
print(f"\n📍 NŒUDS ({len(graph.nodes)} nœuds):")
for i, node in enumerate(graph.nodes.keys(), 1):
    print(f"  {i}. {node}")

# Afficher les edges
print(f"\n🔗 CONNEXIONS:")
for source, targets in graph.edges.items():
    if isinstance(targets, list):
        for target in targets:
            print(f"  {source} → {target}")
    else:
        print(f"  {source} → {targets}")

print("\n" + "="*80)
print("FLUX PRINCIPAL:")
print("="*80)
print("""
1. classify       → Classifier l'intent de la question
2. plan           → Décider de la stratégie (LOCAL/WEB/BOTH)
3. retrieve       → Récupérer documents locaux (ChromaDB)
   OR web_search  → Rechercher sur le web (DuckDuckGo)
   OR combine     → Combiner RAG + Web
4. generate       → Générer la réponse
5. editor         → Review qualité et amélioration
6. verify         → Vérification finale
7. human_approval → Pause pour validation humaine (HITL)
8. finalize       → Finaliser et logger
""")

print("="*80)
print("\n✨ 8 Agents déployés:")
print("  1. ClassifierAgent  - Classification d'intent")
print("  2. PlannerAgent     - Routing intelligent")
print("  3. RetrieverAgent   - RAG local")
print("  4. WebSearchAgent   - Recherche web")
print("  5. GeneratorAgent   - Génération réponses")
print("  6. EditorAgent      - Quality assurance")
print("  7. VerifierAgent    - Vérification")
print("  8. SuggesterAgent   - Suggestions")

print("\n" + "="*80)
