"""
Episodic and preference memory for the Bhutan Opportunity Matching Agent.

Multi-student version:
- Uses 8-digit Student IDs stored as TEXT.
- Keeps interactions and preferences separate by student.
- Scopes feedback updates to the owning student when student_id is supplied.
- Uses safe SQLite connection handling.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DB = PROJECT_ROOT / "data" / "processed" / "agent_memory.db"
MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)


def _validate_student_id(student_id):
    """Return a normalized 8-digit Student ID or raise ValueError."""
    normalized_id = str(student_id).strip()

    if len(normalized_id) != 8 or not normalized_id.isdigit():
        raise ValueError(
            "Student ID must contain exactly 8 digits "
            "(for example: 07230000)."
        )

    return normalized_id


def _get_connection():
    """Create a SQLite connection with row access by column name."""
    connection = sqlite3.connect(MEMORY_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_memory():
    """Create both episodic and preference memory tables."""
    with _get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS interaction_history (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_id TEXT NOT NULL,
                user_request TEXT NOT NULL,
                intent TEXT,
                selected_tool TEXT,
                status TEXT,
                result_summary TEXT,
                feedback TEXT
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_interaction_student
            ON interaction_history(student_id, interaction_id)
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS student_preferences (
                preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                preference_type TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(student_id, preference_type, preference_value)
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_preferences_student
            ON student_preferences(student_id, preference_type)
        """)

    print("Episodic and preference memory initialized successfully.")


def initialize_preference_memory():
    """Backward-compatible initializer for the preference table."""
    initialize_memory()


def save_interaction(
    student_id,
    user_request,
    intent,
    selected_tool,
    status,
    result_summary
):
    """Save an interaction under the selected student's ID."""
    student_id = _validate_student_id(student_id)

    with _get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO interaction_history (
                timestamp,
                student_id,
                user_request,
                intent,
                selected_tool,
                status,
                result_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            student_id,
            str(user_request),
            intent,
            selected_tool,
            status,
            str(result_summary)
        ))

        return cursor.lastrowid


def get_recent_interactions(student_id, limit=5):
    """Retrieve recent interactions belonging only to this student."""
    student_id = _validate_student_id(student_id)

    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    with _get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM interaction_history
            WHERE student_id = ?
            ORDER BY interaction_id DESC
            LIMIT ?
        """, (student_id, limit)).fetchall()

        return [dict(row) for row in rows]


def save_feedback(interaction_id, feedback, student_id=None):
    """
    Save feedback for an interaction.

    Pass student_id to ensure the interaction belongs to that student.
    The optional argument preserves compatibility with older calls, but
    new multi-student code should always pass student_id.
    """
    try:
        interaction_id = int(interaction_id)
    except (TypeError, ValueError):
        return False

    with _get_connection() as connection:
        if student_id is not None:
            student_id = _validate_student_id(student_id)
            cursor = connection.execute("""
                UPDATE interaction_history
                SET feedback = ?
                WHERE interaction_id = ? AND student_id = ?
            """, (str(feedback), interaction_id, student_id))
        else:
            # Backward compatibility only. Prefer passing student_id.
            cursor = connection.execute("""
                UPDATE interaction_history
                SET feedback = ?
                WHERE interaction_id = ?
            """, (str(feedback), interaction_id))

        return cursor.rowcount > 0


def save_student_preference(
    student_id,
    preference_type,
    preference_value
):
    """Save or refresh a preference for one student only."""
    student_id = _validate_student_id(student_id)
    preference_type = str(preference_type).strip()
    preference_value = str(preference_value).strip()

    if not preference_type or not preference_value:
        raise ValueError("Preference type and value cannot be empty.")

    with _get_connection() as connection:
        connection.execute("""
            INSERT INTO student_preferences (
                student_id,
                preference_type,
                preference_value,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(
                student_id,
                preference_type,
                preference_value
            )
            DO UPDATE SET
                updated_at = excluded.updated_at
        """, (
            student_id,
            preference_type,
            preference_value,
            datetime.now().isoformat(timespec="seconds")
        ))

    return True


def get_student_preferences(student_id):
    """Retrieve preferences belonging only to the specified student."""
    student_id = _validate_student_id(student_id)

    with _get_connection() as connection:
        rows = connection.execute("""
            SELECT preference_type, preference_value
            FROM student_preferences
            WHERE student_id = ?
            ORDER BY preference_type, updated_at
        """, (student_id,)).fetchall()

    preferences = {}
    for row in rows:
        preference_type = row["preference_type"]
        preferences.setdefault(preference_type, []).append(
            row["preference_value"]
        )

    return preferences


def learn_from_feedback(student_id, feedback):
    """
    Extract simple preference signals from feedback and save them
    under the supplied student's ID.
    """
    student_id = _validate_student_id(student_id)
    feedback_lower = str(feedback).lower()
    learned_preferences = []

    preference_keywords = {
        "research": ("Interest", "Research"),
        "artificial intelligence": (
            "Interest", "Artificial Intelligence"
        ),
        "data science": ("Interest", "Data Science"),
        "machine learning": (
            "Interest", "Machine Learning"
        ),
        "internship": ("Opportunity type", "Internship"),
        "scholarship": ("Opportunity type", "Scholarship"),
        "bhutan": ("Location", "Bhutan"),
        "international": ("Location", "International"),
        "remote": ("Location", "Remote")
    }

    for keyword, preference in preference_keywords.items():
        if keyword in feedback_lower:
            preference_type, preference_value = preference

            save_student_preference(
                student_id,
                preference_type,
                preference_value
            )

            learned_preferences.append({
                "type": preference_type,
                "value": preference_value
            })

    return learned_preferences


# Initialize both tables when this module is imported.
initialize_memory()
