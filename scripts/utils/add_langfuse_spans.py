#!/usr/bin/env python3
"""
Script pour ajouter automatiquement les spans Langfuse aux nœuds du workflow.
"""

import re

# Lire le fichier
filepath = "/Users/paul/Desktop/Cours M2 /Projet Math/math-rag-system/src/workflow/langgraph_pipeline.py"

with open(filepath, 'r') as f:
    content = f.read()

# Patterns pour les nœuds à instrumenter (qui ne sont pas déjà fait)
nodes_to_instrument = [
    {
        "name": "generate",
        "pattern": r'(def generate_node\(state: WorkflowState, config: Any\) -> WorkflowState:\s+"""[^"]+"""\s+logger\.info\("→ Generate node"\)\s+)(generator: GeneratorAgent)',
        "span_input": '{"question": state["question"][:100], "strategy": state.get("search_strategy")}',
        "span_output": '{"response_length": len(state["generated_answer"]), "sources_count": len(state["sources_cited"])}',
        "insert_before_return": True
    },
    {
        "name": "verify",
        "pattern": r'(def verify_node\(state: WorkflowState, config: Any\) -> WorkflowState:\s+"""[^"]+"""\s+logger\.info\("→ Verify node"\)\s+)(verifier: VerifierAgent)',
        "span_input": '{"question": state["question"][:100]}',
        "span_output": '{"is_valid": state["verification_result"]["is_valid"], "confidence": state["confidence_score"]}',
        "insert_before_return": True
    },
    {
        "name": "editor",
        "pattern": r'(def editor_node\(state: WorkflowState, config: Any\) -> WorkflowState:\s+"""[^"]+"""\s+logger\.info\("→ Editor node"\)\s+)(editor: EditorAgent)',
        "span_input": '{"question": state["question"][:100]}',
        "span_output": '{"quality_score": state["editor_quality_score"], "needs_revision": state["needs_revision"]}',
        "insert_before_return": True
    }
]

for node in nodes_to_instrument:
    # Créer le code de span
    span_create = f'''
    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="{node["name"]}",
        input_data={node["span_input"]}
    )

    '''

    # Insérer le span au début du nœud
    content = re.sub(
        node["pattern"],
        r'\1' + span_create + r'\2',
        content,
        flags=re.MULTILINE | re.DOTALL
    )

    # Ajouter finalize_node_span avant le return du nœud
    if node.get("insert_before_return"):
        # Trouver le return state à la fin du nœud
        node_func_pattern = f'(def {node["name"]}_node.*?)(return state)'

        finalize_code = f'''
    # Finaliser span
    finalize_node_span(span, {node["span_output"]})

    '''

        # On cherche le dernier "return state" dans la fonction
        # Pour être précis, on cherche seulement dans la fonction concernée
        node_start = content.find(f'def {node["name"]}_node')
        if node_start != -1:
            # Trouver la prochaine fonction ou fin de fichier
            next_def = content.find('\ndef ', node_start + 1)
            if next_def == -1:
                next_def = len(content)

            node_section = content[node_start:next_def]

            # Trouver le dernier "return state" dans cette section
            last_return_pos = node_section.rfind('return state')
            if last_return_pos != -1:
                # Insérer le finalize avant le return
                node_section = node_section[:last_return_pos] + finalize_code + node_section[last_return_pos:]
                content = content[:node_start] + node_section + content[next_def:]

# Sauvegarder
with open(filepath, 'w') as f:
    f.write(content)

print("✓ Spans Langfuse ajoutés aux nœuds: generate, verify, editor")
