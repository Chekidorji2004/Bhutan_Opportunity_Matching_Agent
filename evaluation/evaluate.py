"""
Offline evaluation for the Bhutan Student Opportunity Matching Agent.

Run from the project root:
    python -m evaluation.evaluate

This script does NOT call the LLM or consume an API request.
It tests the rule-based tools, profiles, application checklist, deadlines,
and student-specific SQLite memory using a temporary test database.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import opportunity_tools as tools
from src.memory import episodic_memory as memory

TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def result_record(test_id, test_name, passed, expected, actual, details=""):
    return {
        "test_id": test_id,
        "test_name": test_name,
        "result": "PASS" if passed else "FAIL",
        "expected": str(expected),
        "actual": str(actual),
        "details": str(details),
    }


def run_profile_test(row):
    student = tools.get_student_profile(row.student_id)
    passed = student is not None
    return result_record(
        row.test_id, f"Profile exists: {row.student_id}",
        passed, True, passed,
        "Student profile was found." if passed else "Student profile was not found."
    )


def run_search_test(row):
    if row.test_action == "internship_bhutan":
        found = tools.search_opportunities(
            opportunity_type="Internship", location="Bhutan", include_demo=True
        )
    elif row.test_action == "scholarships":
        found = tools.search_opportunities(
            opportunity_type="Scholarship", include_demo=True
        )
    else:
        raise ValueError(f"Unknown search test action: {row.test_action}")

    passed = isinstance(found, pd.DataFrame) and not found.empty
    actual = len(found) if isinstance(found, pd.DataFrame) else "not a DataFrame"
    return result_record(
        row.test_id, row.test_action, passed, "at least 1 result", actual,
        "Search uses the current opportunities.csv data."
    )


def run_eligibility_test(row):
    student = tools.get_student_profile(row.student_id)
    opportunity = tools.get_opportunity(row.input_value)
    if student is None or opportunity is None:
        return result_record(
            row.test_id, f"Eligibility {row.student_id}/{row.input_value}",
            False, row.expected_result, "Student or opportunity not found",
            "Check the IDs in student_profiles.csv and opportunities.csv."
        )

    assessment = tools.check_eligibility(student, opportunity)
    actual = assessment["eligibility_status"]
    passed = actual == row.expected_result
    return result_record(
        row.test_id, f"Eligibility {row.student_id}/{row.input_value}",
        passed, row.expected_result, actual,
        "Failed requirements: " + "; ".join(assessment["failed_requirements"])
        + " | Missing information: " + "; ".join(assessment["missing_information"])
    )


def run_ranking_test(row):
    student = tools.get_student_profile(row.student_id)
    data = tools.opportunities.copy()
    ranked = tools.rank_opportunities(student, data)

    scores_valid = (
        not ranked.empty
        and ranked["match_score"].between(0, 100).all()
    )

    priority = {"ELIGIBLE": 0, "NEEDS VERIFICATION": 1, "NOT ELIGIBLE": 2}
    priority_values = ranked["eligibility"].map(priority).tolist()
    eligibility_sorted = priority_values == sorted(priority_values)

    # Within each eligibility group, match score should be descending.
    score_sorted = True
    for status in ranked["eligibility"].dropna().unique():
        group = ranked.loc[ranked["eligibility"] == status, "match_score"].tolist()
        if group != sorted(group, reverse=True):
            score_sorted = False
            break

    passed = scores_valid and eligibility_sorted and score_sorted
    return result_record(
        row.test_id, f"Ranking for {row.student_id}",
        passed, "scores 0-100; eligibility priority and scores descending",
        f"scores_valid={scores_valid}; eligibility_sorted={eligibility_sorted}; "
        f"score_sorted={score_sorted}; rows={len(ranked)}",
        "Ranking is deterministic and based on the configured weighted rules."
    )


def run_application_test(row):
    plan = tools.create_application_plan(
        row.input_value,
        available_documents=[],
        student_id=row.student_id
    )
    passed = bool(plan.get("success")) and isinstance(plan.get("checklist"), pd.DataFrame)
    readiness = plan.get("readiness")
    passed = passed and isinstance(readiness, (int, float)) and 0 <= readiness <= 100
    return result_record(
        row.test_id, f"Application plan {row.input_value} for {row.student_id}",
        passed, "success with checklist and readiness between 0 and 100",
        f"success={plan.get('success')}; readiness={readiness}; "
        f"checklist_rows={len(plan.get('checklist', [])) if isinstance(plan.get('checklist'), pd.DataFrame) else 'n/a'}",
        "A readiness percentage measures document preparation, not selection probability."
    )


def run_deadline_test(row):
    fixed_today = date(2026, 11, 14)
    cases = [
        (fixed_today - timedelta(days=1), "EXPIRED"),
        (fixed_today + timedelta(days=3), "URGENT"),
        (fixed_today + timedelta(days=20), "HIGH"),
        (fixed_today + timedelta(days=45), "MEDIUM"),
        (fixed_today + timedelta(days=90), "LOW"),
    ]
    actuals = [
        tools.calculate_deadline_priority(d.isoformat(), fixed_today)["priority"]
        for d, _ in cases
    ]
    expected = [expected for _, expected in cases]
    unknown = tools.calculate_deadline_priority("not-a-date", fixed_today)["priority"]
    passed = actuals == expected and unknown == "UNKNOWN"
    return result_record(
        row.test_id, "Deadline classification",
        passed, expected + ["UNKNOWN"], actuals + [unknown],
        "Fixed reference date: 2026-11-14, so results do not change with the day you run the test."
    )



def run_memory_isolation_test(row):
    """
    Test that different students have separate interaction histories
    and preferences.

    Uses a uniquely named SQLite database in evaluation/results.
    It does not modify the real agent_memory.db.
    """

    import uuid

    original_db = memory.MEMORY_DB

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Create a unique test database.
        # We intentionally keep this file to avoid Windows file-lock
        # errors when SQLite connections are still open.
        test_db = RESULTS_DIR / (
            f"memory_isolation_test_{uuid.uuid4().hex[:8]}.db"
        )

        memory.MEMORY_DB = test_db
        memory.initialize_memory()

        id_a = "07230000"
        id_b = "07230001"

        # Save separate interactions.
        memory.save_interaction(
            id_a,
            "TEST request A",
            "search_opportunities",
            "search_tool",
            "success",
            "Student A test record"
        )

        memory.save_interaction(
            id_b,
            "TEST request B",
            "check_eligibility",
            "eligibility_tool",
            "success",
            "Student B test record"
        )

        # Retrieve individual histories.
        history_a = memory.get_recent_interactions(
            id_a, limit=10
        )

        history_b = memory.get_recent_interactions(
            id_b, limit=10
        )

        # Save separate preferences.
        memory.save_student_preference(
            id_a, "Interest", "TEST_A"
        )

        preferences_a = memory.get_student_preferences(id_a)
        preferences_b = memory.get_student_preferences(id_b)

        # Verify isolation.
        passed = (
            len(history_a) == 1
            and len(history_b) == 1
            and history_a[0]["student_id"] == id_a
            and history_b[0]["student_id"] == id_b
            and "TEST_A" in preferences_a.get("Interest", [])
            and "TEST_A" not in preferences_b.get("Interest", [])
        )

        return result_record(
            row.test_id,
            "Student memory isolation",
            passed,
            "separate histories and preferences",
            (
                f"A_history={len(history_a)}; "
                f"B_history={len(history_b)}; "
                f"A_preferences={preferences_a}; "
                f"B_preferences={preferences_b}"
            ),
            "Uses a separate test database. "
            "The real student memory database is unchanged."
        )

    except Exception:
        raise

    finally:
        # Restore the original database path.
        memory.MEMORY_DB = original_db



def run_one(row):
    actions = {
        "profile_exists": run_profile_test,
        "internship_bhutan": run_search_test,
        "scholarships": run_search_test,
        "eligibility": run_eligibility_test,
        "ranking": run_ranking_test,
        "application_plan": run_application_test,
        "deadline_rules": run_deadline_test,
        "memory_isolation": run_memory_isolation_test,
    }
    function = actions.get(row.test_action)
    if function is None:
        return result_record(
            row.test_id, row.test_action, False, "known test action",
            "unknown action", "Add a handler in evaluate.py."
        )
    try:
        return function(row)
    except Exception as exc:
        return result_record(
            row.test_id, row.test_action, False, "test completes without exception",
            f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)
        )


def main():
    if not TEST_CASES_PATH.exists():
        raise FileNotFoundError(f"Missing test cases: {TEST_CASES_PATH}")

    test_cases = pd.read_csv(TEST_CASES_PATH, dtype={"student_id": str}).fillna("")
    results = [run_one(row) for row in test_cases.itertuples(index=False)]
    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "offline_test_results.csv"
    json_path = RESULTS_DIR / "offline_test_summary.json"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    total = len(results_df)
    passed = int((results_df["result"] == "PASS").sum())
    failed = total - passed
    pass_rate = round((passed / total) * 100, 2) if total else 0

    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_percent": pass_rate,
        "api_calls_used": 0,
        "test_date": date.today().isoformat(),
        "results_csv": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    json_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\n" + "=" * 64)
    print("BHUTAN STUDENT OPPORTUNITY AGENT - OFFLINE EVALUATION")
    print("=" * 64)
    print(results_df[["test_id", "test_name", "result"]].to_string(index=False))
    print("-" * 64)
    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Pass rate   : {pass_rate}%")
    print("API calls   : 0")
    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")

    if failed:
        print("\nSome tests failed. Fix the issue and run the evaluation again.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
