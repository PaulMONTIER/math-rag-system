#!/usr/bin/env python3
"""
Script pour ajouter les spans Langfuse aux nœuds restants.
"""

# Ce script génère les modifications à faire pour generate_node, editor_node et verify_node

nodes = [
    {
        "name": "generate",
        "input": '{"question": state["question"][:100], "strategy": state.get("search_strategy")}',
        "output": '{"response_length": len(state["generated_answer"]), "sources_count": len(state["sources_cited"])}'
    },
    {
        "name": "editor",
        "input": '{"question": state["question"][:100]}',
        "output": '{"quality_score": state["editor_quality_score"], "needs_revision": state["needs_revision"]}'
    },
    {
        "name": "verify",
        "input": '{"question": state["question"][:100]}',
        "output": '{"is_valid": state["verification_result"]["is_valid"], "confidence": state["confidence_score"]}'
    }
]

for node in nodes:
    print(f"\n# {node['name']}_node:")
    print(f"""
    # Créer span Langfuse
    span = create_node_span(
        trace=state.get("langfuse_trace"),
        node_name="{node['name']}",
        input_data={node['input']}
    )
    """)

    print(f"""
    # ... (code existant) ...

    # Finaliser span
    finalize_node_span(span, {node['output']})
    """)
