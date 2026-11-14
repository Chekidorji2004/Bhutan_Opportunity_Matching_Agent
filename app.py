"""Streamlit application for the Bhutan Student Opportunity Matching Agent.

Multi-student demo application. Student ID selection separates displayed
profiles, chat sessions, application plans, feedback, and episodic memory.
This is a demo selector, not secure authentication.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration must be called before other Streamlit commands.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bhutan Opportunity Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project modules
from src.agent.agent_workflow import ask_agent
from src.memory.episodic_memory import (
    initialize_memory,
    initialize_preference_memory,
    get_recent_interactions,
    get_student_preferences,
    save_feedback,
)
from src.tools.opportunity_tools import (
    create_application_plan,
    get_student_profile,
    opportunities,
    split_values,
)

# -----------------------------------------------------------------------------
# Paths and database setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "data" / "student_profiles.csv"
DB_PATH = BASE_DIR / "data" / "processed" / "agent_memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

initialize_memory()
initialize_preference_memory()


def initialize_application_plan_table():
    """Create persistent application-plan storage, scoped to each student."""
    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS application_plans (
                plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                opportunity_id TEXT NOT NULL,
                opportunity_title TEXT NOT NULL,
                checklist_json TEXT NOT NULL,
                readiness REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(student_id, opportunity_id)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_application_plans_student
            ON application_plans(student_id, plan_id)
        """)


initialize_application_plan_table()


def load_student_profiles():
    if not PROFILE_PATH.exists():
        st.error(f"Student profile file not found: {PROFILE_PATH}")
        st.stop()

    frame = pd.read_csv(PROFILE_PATH, dtype={"student_id": str}).fillna("")
    required = {"student_id", "name"}
    missing = required - set(frame.columns)
    if missing:
        st.error(f"student_profiles.csv is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    frame["student_id"] = frame["student_id"].astype(str).str.strip()
    frame = frame[frame["student_id"].str.fullmatch(r"\d{8}", na=False)]
    if frame.empty:
        st.error("No valid 8-digit Student IDs were found in student_profiles.csv.")
        st.stop()
    return frame


student_profiles = load_student_profiles()


def get_plan_rows(student_id):
    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT * FROM application_plans
            WHERE student_id = ?
            ORDER BY updated_at DESC, plan_id DESC
        """, (student_id,)).fetchall()
        return [dict(row) for row in rows]


def save_plan(student_id, opportunity_id, title, checklist):
    """Insert or update one student's plan for an opportunity."""
    now = datetime.now().isoformat(timespec="seconds")
    total = len(checklist)
    completed = sum(1 for item in checklist if item.get("status") == "COMPLETED")
    readiness = round((completed / total) * 100, 2) if total else 100.0
    checklist_json = json.dumps(checklist, ensure_ascii=False)

    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.execute("""
            INSERT INTO application_plans (
                student_id, opportunity_id, opportunity_title,
                checklist_json, readiness, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id, opportunity_id)
            DO UPDATE SET
                opportunity_title = excluded.opportunity_title,
                checklist_json = excluded.checklist_json,
                readiness = excluded.readiness,
                updated_at = excluded.updated_at
        """, (
            student_id, str(opportunity_id), str(title),
            checklist_json, readiness, now, now
        ))
    return readiness


def delete_plan(student_id, plan_id):
    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        cursor = connection.execute(
            "DELETE FROM application_plans WHERE plan_id = ? AND student_id = ?",
            (int(plan_id), student_id),
        )
        return cursor.rowcount > 0


def safe_text(value, fallback="Not provided"):
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def display_semicolon_values(value):
    values = split_values(value)
    if not values:
        st.write("Not provided")
    else:
        st.write(" · ".join(item.title() for item in values))


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 750; color: #176B55; }
    .subtitle { font-size: 1rem; color: #777; }
    .feature-card {
        padding: 18px; border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3); min-height: 125px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 12px; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Keep separate chat transcripts for each selected student in this browser session.
if "student_messages" not in st.session_state:
    st.session_state.student_messages = {}
if "active_student_id" not in st.session_state:
    st.session_state.active_student_id = None

# -----------------------------------------------------------------------------
# Sidebar: student selector and navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Opportunity Agent")
    st.subheader("Student Access")

    student_ids = student_profiles["student_id"].tolist()
    selected_student_id = st.selectbox(
        "Select your Student ID",
        options=student_ids,
        format_func=lambda sid: f"{sid} — {student_profiles.loc[student_profiles['student_id'] == sid, 'name'].iloc[0]}",
        key="selected_student_id",
    )

    selected_profile = student_profiles.loc[
        student_profiles["student_id"] == selected_student_id
    ].iloc[0]
    st.success(f"Welcome, {selected_profile['name']}!")
    st.caption("Demo access: selecting an ID changes the student view. This is not secure login/authentication.")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "AI Assistant", "My Profile", "My Application Plans", "Interaction History"],
        key="navigation_page",
    )

# When a different student is selected, their separate chat transcript is loaded.
if st.session_state.active_student_id != selected_student_id:
    st.session_state.active_student_id = selected_student_id

messages_by_student = st.session_state.student_messages
messages_by_student.setdefault(selected_student_id, [])
current_messages = messages_by_student[selected_student_id]

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="main-title">Bhutan Opportunity Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discover internships and scholarships that match your goals.</div>', unsafe_allow_html=True)
    st.write("")
    st.info(
        f"Welcome, {selected_profile['name']}. Your assistant can help you discover opportunities, "
        "check eligibility and prepare application documents."
    )

    total_opportunities = len(opportunities)
    demo_count = 0
    if "is_demo" in opportunities.columns:
        demo_count = int(opportunities["is_demo"].astype(str).str.lower().eq("true").sum())
    student_plans = get_plan_rows(selected_student_id)
    interactions = get_recent_interactions(selected_student_id, limit=1000)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Available opportunity records", total_opportunities)
    metric2.metric("My application plans", len(student_plans))
    metric3.metric("My interactions", len(interactions))

    if demo_count:
        st.warning(f"{demo_count} of the current opportunity records are fictional demonstration records. Verify real opportunities from official sources before applying.")

    st.subheader("What you can do")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="feature-card"><h3>🔍 Find Opportunities</h3><p>Search for internships and scholarships based on your interests.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><h3>📋 Application Planning</h3><p>Create document checklists and track your preparation progress.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><h3>✅ Check Eligibility</h3><p>Understand which listed requirements you meet or need to verify.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><h3>🧠 Personalised Matching</h3><p>Use your academic profile, skills and preferences to rank opportunities.</p></div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Start Exploring Opportunities", type="primary"):
        st.session_state.navigation_page = "AI Assistant"
        st.rerun()

# -----------------------------------------------------------------------------
# AI ASSISTANT
# -----------------------------------------------------------------------------
elif page == "AI Assistant":
    st.title("🤖 AI Opportunity Assistant")
    st.write("Ask questions about internships, scholarships, eligibility and application planning.")
    st.info(f"Currently accessing as: {selected_student_id} ({selected_profile['name']})")
    st.caption("Example: Find me suitable data science internships in Bhutan.")

    top_left, top_right = st.columns([5, 1])
    with top_right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.student_messages[selected_student_id] = []
            st.rerun()

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask your opportunity assistant...", key=f"chat_input_{selected_student_id}")
    if user_prompt:
        current_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Your AI agent is thinking..."):
                try:
                    result = ask_agent(user_prompt, student_id=selected_student_id)
                    if isinstance(result, dict):
                        final_response = result.get("final_response") or result.get("response") or str(result)
                    else:
                        final_response = str(result)
                    st.markdown(final_response)
                    current_messages.append({"role": "assistant", "content": final_response})
                except Exception as error:
                    error_message = f"Agent error: {error}"
                    st.error(error_message)
                    current_messages.append({"role": "assistant", "content": error_message})

# -----------------------------------------------------------------------------
# MY PROFILE
# -----------------------------------------------------------------------------
elif page == "My Profile":
    st.title("👤 My Student Profile")
    st.write("Your academic background, skills and career preferences.")

    # Use the currently selected profile, not a hard-coded student ID.
    profile = selected_profile
    st.caption(f"Student ID: {selected_student_id}")

    st.subheader("Personal and Academic Information")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Name:** {safe_text(profile.get('name'))}")
    c1.write(f"**Citizenship:** {safe_text(profile.get('citizenship'))}")
    c2.write(f"**Education:** {safe_text(profile.get('education_level'))}")
    c2.write(f"**Degree:** {safe_text(profile.get('degree'))}")
    c3.write(f"**Year of study:** {safe_text(profile.get('year_of_study'))}")
    c3.metric("GPA", safe_text(profile.get("gpa")))

    st.divider()
    st.subheader("Field of Study")
    st.write(safe_text(profile.get("field_of_study")))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Skills")
        display_semicolon_values(profile.get("skills", ""))
    with c2:
        st.subheader("Interests")
        display_semicolon_values(profile.get("interests", ""))

    st.divider()
    st.subheader("Opportunity Preferences")
    c1, c2, c3 = st.columns(3)
    c1.write("**Opportunity type**")
    c1.write(safe_text(profile.get("preferred_opportunity_type")))
    c2.write("**Preferred locations**")
    c2.write(safe_text(profile.get("preferred_location")))
    c3.write("**Preferred mode**")
    c3.write(safe_text(profile.get("preferred_mode")))

    st.subheader("Career Goal")
    st.info(safe_text(profile.get("career_goal")))

    learned_preferences = get_student_preferences(selected_student_id)
    with st.expander("Learned preferences from feedback"):
        if learned_preferences:
            for preference_type, values in learned_preferences.items():
                st.write(f"**{preference_type}:** {', '.join(values)}")
        else:
            st.caption("No learned preferences have been recorded for this student yet.")

# -----------------------------------------------------------------------------
# MY APPLICATION PLANS
# -----------------------------------------------------------------------------
elif page == "My Application Plans":
    st.title("📋 My Application Plans")
    st.write(f"Application plans saved for **{selected_profile['name']}** ({selected_student_id}).")

    st.subheader("Create an application plan")
    if opportunities.empty:
        st.warning("There are no opportunity records available yet.")
    else:
        opportunity_options = opportunities[["opportunity_id", "title"]].drop_duplicates()
        option_map = {
            f"{row.opportunity_id} — {row.title}": str(row.opportunity_id)
            for row in opportunity_options.itertuples(index=False)
        }
        selected_option = st.selectbox("Choose an opportunity", list(option_map.keys()), key=f"plan_opportunity_{selected_student_id}")
        if st.button("Generate / Refresh Checklist", type="primary"):
            opportunity_id = option_map[selected_option]
            try:
                plan_result = create_application_plan(
                    opportunity_id=opportunity_id,
                    available_documents=[],
                    student_id=selected_student_id,
                )
                if not plan_result.get("success"):
                    st.error(plan_result.get("message", "Could not create the application plan."))
                else:
                    checklist_df = plan_result.get("checklist", pd.DataFrame())
                    checklist = checklist_df.to_dict(orient="records") if isinstance(checklist_df, pd.DataFrame) else list(checklist_df)
                    # Normalize checklist entries for storage and display.
                    normalized_checklist = [
                        {"document": str(item.get("document", "Document")),
                         "status": str(item.get("status", "PENDING")).upper()}
                        for item in checklist
                    ]
                    readiness = save_plan(
                        selected_student_id,
                        opportunity_id,
                        plan_result.get("opportunity", selected_option),
                        normalized_checklist,
                    )
                    st.success(f"Checklist saved. Current readiness: {readiness:.1f}%")
                    st.rerun()
            except Exception as error:
                st.error(f"Could not create application plan: {error}")

    st.divider()
    st.subheader("My saved plans")
    plans = get_plan_rows(selected_student_id)
    if not plans:
        st.info("You have no saved application plans yet. Create one above to get started.")
    else:
        for plan in plans:
            with st.expander(f"{plan['opportunity_title']} — {plan['readiness']:.1f}% ready", expanded=False):
                st.caption(f"Opportunity ID: {plan['opportunity_id']} · Last updated: {plan['updated_at']}")
                try:
                    checklist = json.loads(plan["checklist_json"])
                except (TypeError, json.JSONDecodeError):
                    checklist = []

                updated_checklist = []
                if checklist:
                    st.write("Tick documents you have prepared:")
                    for index, item in enumerate(checklist):
                        is_complete = item.get("status") == "COMPLETED"
                        checked = st.checkbox(
                            item.get("document", f"Document {index + 1}"),
                            value=is_complete,
                            key=f"plan_{plan['plan_id']}_doc_{index}",
                        )
                        updated_checklist.append({
                            "document": item.get("document", f"Document {index + 1}"),
                            "status": "COMPLETED" if checked else "PENDING",
                        })
                else:
                    st.caption("This plan has no required documents in its checklist.")

                if st.button("Save checklist progress", key=f"save_plan_{plan['plan_id']}"):
                    new_readiness = save_plan(
                        selected_student_id,
                        plan["opportunity_id"],
                        plan["opportunity_title"],
                        updated_checklist,
                    )
                    st.success(f"Progress saved. Readiness: {new_readiness:.1f}%")
                    st.rerun()

                # Show source/application links only when present.
                opportunity_row = opportunities[
                    opportunities["opportunity_id"].astype(str) == str(plan["opportunity_id"])
                ]
                if not opportunity_row.empty:
                    source_url = str(opportunity_row.iloc[0].get("source_url", "")).strip()
                    application_url = str(opportunity_row.iloc[0].get("application_url", "")).strip()
                    if source_url and source_url.lower() != "nan":
                        st.markdown(f"[Opportunity source]({source_url})")
                    if application_url and application_url.lower() != "nan":
                        st.markdown(f"[Application page]({application_url})")
                    if not source_url and not application_url:
                        st.caption("No source or application URL is recorded for this opportunity. Verify details independently; demo records are fictional.")

                if st.button("Delete this plan", key=f"delete_plan_{plan['plan_id']}"):
                    delete_plan(selected_student_id, plan["plan_id"])
                    st.success("Application plan deleted.")
                    st.rerun()

# -----------------------------------------------------------------------------
# INTERACTION HISTORY
# -----------------------------------------------------------------------------
elif page == "Interaction History":
    st.title("🕒 My Interaction History")
    st.write(f"Showing conversations and requests for **{selected_profile['name']}** ({selected_student_id}).")

    history = get_recent_interactions(selected_student_id, limit=100)
    if not history:
        st.info("No interaction history has been recorded for this student yet. Start a conversation in AI Assistant.")
    else:
        st.metric("Recorded interactions", len(history))
        for interaction in history:
            interaction_id = interaction.get("interaction_id")
            timestamp = interaction.get("timestamp", "")
            request = interaction.get("user_request", "")
            status = interaction.get("status", "")
            intent = interaction.get("intent", "")
            selected_tool = interaction.get("selected_tool", "")
            result_summary = interaction.get("result_summary", "")
            feedback = interaction.get("feedback", "")

            with st.expander(f"{timestamp} — {str(request)[:90]}"):
                st.write("**Request:**", request)
                st.write("**Intent:**", safe_text(intent))
                st.write("**Selected tool:**", safe_text(selected_tool))
                st.write("**Status:**", safe_text(status))
                st.write("**Result summary:**", safe_text(result_summary))
                st.write("**Existing feedback:**", safe_text(feedback))

                feedback_text = st.text_input(
                    "Add or update feedback",
                    value="" if not feedback else str(feedback),
                    key=f"feedback_{selected_student_id}_{interaction_id}",
                )
                if st.button("Save feedback", key=f"save_feedback_{selected_student_id}_{interaction_id}"):
                    try:
                        updated = save_feedback(
                            interaction_id=interaction_id,
                            feedback=feedback_text,
                            student_id=selected_student_id,
                        )
                        if updated:
                            st.success("Feedback saved.")
                            st.rerun()
                        else:
                            st.error("Feedback was not saved. The interaction may not belong to this student.")
                    except Exception as error:
                        st.error(f"Could not save feedback: {error}")

    if st.button("Refresh history"):
        st.rerun()

st.sidebar.divider()
st.sidebar.caption("Bhutan Student Opportunity Matching Agent · Multi-student demonstration")
