"""
Script d'exécution des questions de test.

Charge les questions depuis test_questions.json et les exécute
à travers le workflow complet pour valider le système.

Usage:
    python tests/run_test_questions.py
    python tests/run_test_questions.py --limit 5  # Tester seulement 5 questions
    python tests/run_test_questions.py --category analyse  # Une seule catégorie
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow.langgraph_pipeline import create_rag_workflow, invoke_workflow
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_test_questions(file_path: Path) -> Dict:
    """Charge les questions de test depuis JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_latex_preservation(text: str) -> Dict:
    """Vérifie que les formules LaTeX sont présentes et non coupées."""
    import re

    # Patterns LaTeX
    patterns = {
        'inline_dollar': r'\$[^\$]+\$',
        'display_dollar': r'\$\$[^\$]+\$\$',
        'equation': r'\\begin\{equation\}.*?\\end\{equation\}',
        'bracket': r'\\\[.*?\\\]'
    }

    found_formulas = []
    total_count = 0

    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.DOTALL)
        total_count += len(matches)
        if matches:
            found_formulas.extend([{"type": name, "formula": m} for m in matches])

    # Vérifier qu'aucune formule n'est coupée (heuristique simple)
    incomplete_patterns = [
        r'\$[^\$]*$',  # $ sans fermeture
        r'\\begin\{[^}]+\}(?!.*\\end)',  # \begin sans \end
    ]

    has_incomplete = any(re.search(p, text) for p in incomplete_patterns)

    return {
        "total_formulas": total_count,
        "formulas": found_formulas[:5],  # Seulement 5 exemples
        "has_incomplete": has_incomplete
    }


def validate_response(
    question: Dict,
    result: Dict,
    criteria: Dict
) -> Dict:
    """Valide une réponse selon les critères."""
    validation = {
        "question_id": question["id"],
        "passed": True,
        "checks": {}
    }

    response = result.get("final_response", "")

    # 1. Longueur
    min_len = criteria["response_quality"]["min_length"]
    max_len = criteria["response_quality"]["max_length"]
    length_ok = min_len <= len(response) <= max_len
    validation["checks"]["length"] = {
        "passed": length_ok,
        "value": len(response),
        "expected": f"{min_len}-{max_len}"
    }

    # 2. Sources citées
    sources = result.get("sources_cited", [])
    has_sources = len(sources) > 0
    validation["checks"]["sources_cited"] = {
        "passed": has_sources,
        "value": len(sources),
        "expected": ">= 1"
    }

    # 3. LaTeX préservé (si attendu)
    if question.get("should_contain_formulas", False):
        latex_check = check_latex_preservation(response)
        latex_ok = latex_check["total_formulas"] > 0 and not latex_check["has_incomplete"]
        validation["checks"]["latex_preservation"] = {
            "passed": latex_ok,
            "value": latex_check["total_formulas"],
            "has_incomplete": latex_check["has_incomplete"]
        }

    # 4. Confiance
    confidence = result.get("confidence_score", 0)
    threshold = criteria["generation_quality"]["confidence_threshold"]
    confidence_ok = confidence >= threshold
    validation["checks"]["confidence"] = {
        "passed": confidence_ok,
        "value": confidence,
        "expected": f">= {threshold}"
    }

    # 5. Succès global
    validation["passed"] = result.get("success", False) and all(
        check["passed"] for check in validation["checks"].values()
    )

    return validation


def generate_report(
    results: List[Dict],
    output_path: Path
):
    """Génère un rapport de test."""
    total = len(results)
    passed = sum(1 for r in results if r["validation"]["passed"])
    failed = total - passed

    # Statistiques par check
    check_stats = {}
    for result in results:
        for check_name, check_data in result["validation"]["checks"].items():
            if check_name not in check_stats:
                check_stats[check_name] = {"passed": 0, "failed": 0}

            if check_data["passed"]:
                check_stats[check_name]["passed"] += 1
            else:
                check_stats[check_name]["failed"] += 1

    # Créer rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0
        },
        "check_statistics": check_stats,
        "results": results
    }

    # Sauvegarder JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved to: {output_path}")

    return report


def print_summary(report: Dict):
    """Affiche un résumé du rapport."""
    print("\n" + "═" * 80)
    print("  TEST SUMMARY")
    print("═" * 80)
    print()

    summary = report["summary"]
    print(f"Total questions: {summary['total_questions']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed: {summary['failed']}")
    print()

    # Statistiques par check
    print("Check statistics:")
    for check_name, stats in report["check_statistics"].items():
        total_check = stats["passed"] + stats["failed"]
        pass_rate = stats["passed"] / total_check if total_check > 0 else 0
        status = "✓" if pass_rate >= 0.8 else "⚠️" if pass_rate >= 0.5 else "❌"
        print(f"  {status} {check_name}: {stats['passed']}/{total_check} ({pass_rate:.1%})")
    print()

    # Questions échouées
    failed_results = [r for r in report["results"] if not r["validation"]["passed"]]
    if failed_results:
        print("Failed questions:")
        for result in failed_results[:5]:  # Max 5
            q = result["question"]
            print(f"  ❌ Q{q['id']}: {q['question'][:60]}...")
            # Raisons
            for check_name, check_data in result["validation"]["checks"].items():
                if not check_data["passed"]:
                    print(f"     → {check_name}: {check_data.get('value', 'N/A')} (expected: {check_data.get('expected', 'N/A')})")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")
    print()


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Run test questions through RAG workflow"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to test"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Test only questions from this category"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/logs/test_report.json",
        help="Output report path (default: data/logs/test_report.json)"
    )

    args = parser.parse_args()

    # Banner
    print("═" * 80)
    print("  TEST QUESTIONS - VALIDATION DU SYSTÈME RAG")
    print("═" * 80)
    print()

    try:
        # 1. Charger questions de test
        test_file = Path(__file__).parent / "test_questions.json"
        logger.info(f"Loading test questions from: {test_file}")

        test_data = load_test_questions(test_file)
        questions = test_data["test_questions"]
        criteria = test_data["validation_criteria"]

        # Filtrer par catégorie si demandé
        if args.category:
            questions = [q for q in questions if q["category"] == args.category]
            print(f"Filtering by category: {args.category}")

        # Limiter nombre si demandé
        if args.limit:
            questions = questions[:args.limit]
            print(f"Limiting to {args.limit} questions")

        print(f"Total questions to test: {len(questions)}")
        print()

        # 2. Initialiser système
        print("Initializing RAG system...")
        config = load_config()
        workflow = create_rag_workflow(config)
        print("✓ System initialized")
        print()

        # 3. Exécuter questions
        print("Running test questions...")
        print()

        results = []

        for question in tqdm(questions, desc="Testing"):
            logger.info(f"Testing Q{question['id']}: {question['question']}")

            # Invoquer workflow
            result = invoke_workflow(
                workflow=workflow,
                question=question["question"],
                student_level=question["level"]
            )

            # Valider réponse
            validation = validate_response(question, result, criteria)

            # Stocker résultat
            results.append({
                "question": question,
                "result": {
                    "success": result["success"],
                    "final_response": result["final_response"][:500],  # Tronquer
                    "sources_cited": result.get("sources_cited", []),
                    "confidence_score": result.get("confidence_score", 0),
                    "metadata": result.get("metadata", {})
                },
                "validation": validation
            })

        print()

        # 4. Générer rapport
        print("Generating report...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = generate_report(results, output_path)
        print(f"✓ Report saved to: {output_path}")

        # 5. Afficher résumé
        print_summary(report)

        # 6. Exit code
        if report["summary"]["pass_rate"] < 0.7:
            print("⚠️  Warning: Pass rate below 70%")
            sys.exit(1)
        else:
            print("✓ Test suite passed!")

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        logger.info("Test cancelled by user")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
        print("   Check logs for details: data/logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# NOTES DÉVELOPPEUR
# ═══════════════════════════════════════════════════════════════════════════════
#
# OBJECTIF:
# Valider automatiquement le système RAG avec un ensemble de questions de test.
# Générer rapport détaillé avec métriques et résultats.
#
# WORKFLOW:
# 1. Charger test_questions.json
# 2. Filtrer par catégorie/limite si demandé
# 3. Pour chaque question:
#    - Invoquer workflow complet
#    - Valider réponse selon critères
#    - Stocker résultats
# 4. Générer rapport JSON
# 5. Afficher résumé
#
# VALIDATIONS:
# - Longueur réponse (50-2000 chars)
# - Sources citées (>= 1)
# - LaTeX préservé (si should_contain_formulas)
# - Confiance >= seuil (0.75)
# - Succès global
#
# RAPPORT:
# - JSON avec tous les résultats
# - Statistiques par check
# - Liste des échecs
# - Pass rate global
#
# USAGE:
# ```bash
# # Tester toutes les questions
# python tests/run_test_questions.py
#
# # Tester seulement 5 questions
# python tests/run_test_questions.py --limit 5
#
# # Tester seulement analyse
# python tests/run_test_questions.py --category analyse
#
# # Custom output
# python tests/run_test_questions.py --output my_report.json
# ```
#
# EXIT CODES:
# - 0: Pass rate >= 70%
# - 1: Pass rate < 70% ou erreur
#
# INTÉGRATION CI/CD:
# - Peut être utilisé dans pipeline CI
# - Exit code indique succès/échec
# - Rapport JSON parseable
#
# ═══════════════════════════════════════════════════════════════════════════════
