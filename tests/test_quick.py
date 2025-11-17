#!/usr/bin/env python3
"""Test rapide du workflow après fix"""

from src.utils.config_loader import load_config
from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow

print('Création du workflow...')
config = load_config()
app = create_rag_workflow(config)
print('✓ Workflow créé\n')

print('Test simple: "Qu\'est-ce qu\'une dérivée?"')
result = invoke_workflow(app, 'Qu est-ce qu une dérivée?', student_level='L2', rigor_level=3)

print(f'\n✅ Succès: {result["success"]}')
print(f'Stratégie utilisée: {result.get("search_strategy", "N/A")}')
print(f'Score qualité Editor: {result.get("editor_quality_score", "N/A")}')
print(f'\nRéponse (200 premiers caractères):\n{result["final_response"][:200]}...')
