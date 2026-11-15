"""
LangGraph agent workflow for the Bhutan Opportunity Matching Agent.

Moved from Bhutan_Opportunity_Agent.ipynb.
"""

import json
import os
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from src.memory.episodic_memory import (
    get_recent_interactions,
    save_interaction
)
from src.tools.opportunity_tools import (
    add_deadline_information,
    check_eligibility,
    create_adapted_student_profile,
    create_application_plan,
    get_opportunity,
    get_student_profile,
    rank_opportunities,
    run_opportunity_agent,
    search_opportunities
)

# Load the OpenRouter API key
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OpenRouter API key not found.")

# Initialise OpenRouter client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)




# ---------------------------------------------------------------------
# Request understanding
# ---------------------------------------------------------------------

def understand_student_request(user_message, selected_student_id):

    system_prompt = """
    You are the request understanding component of a
    Bhutan Student Internship and Scholarship Matching Agent.

    Convert the student's request into JSON.

    Use exactly these fields:

    {
        "intent": "search_opportunities",
        "keyword": null,
        "opportunity_type": null,
        "location": null,
        "student_id": null,
        "opportunity_id": null
    }

    AVAILABLE INTENTS:

    1. search_opportunities
       The student wants to browse or find opportunities.

    2. match_opportunities
       The student wants personalised recommendations.

    3. check_eligibility
       The student wants to know whether they qualify.

    4. application_plan
       The student wants an application checklist or plan.

    5. deadline_check
       The student wants to check upcoming or urgent deadlines.

    6. general_question
       The student asks a general question.

    7. multi_step_planning
       The student asks for several connected tasks in one
       request, such as finding opportunities, checking
       eligibility, ranking them and preparing an
       application plan.
       Use individual intents when only one task is requested.

    EXTRACTION RULES:

    - Extract the subject or field as keyword.
    - Extract the requested location.
    - Opportunity types must be Internship or Scholarship.
    - If scholarships are requested, use Scholarship.
    - If internships are requested, use Internship.
    - A college Student ID contains exactly 8 digits, such as 07230000.
    - An opportunity ID starts with DEMO, such as DEMO001.
    - Never put a DEMO ID in student_id.
    - Never put an 8-digit Student ID in opportunity_id.
    - Use null for information not provided.
    - Never invent IDs or student information.
    - Return valid JSON only.
    - Do not use Markdown code fences.

    EXAMPLES:

    User: Show me available scholarships.
    {
        "intent": "search_opportunities",
        "keyword": null,
        "opportunity_type": "Scholarship",
        "location": null,
        "student_id": null,
        "opportunity_id": null
    }

    User: Find suitable data science internships in Bhutan.
    {
        "intent": "match_opportunities",
        "keyword": "data science",
        "opportunity_type": "Internship",
        "location": "Bhutan",
        "student_id": null,
        "opportunity_id": null
    }

    User: Prepare an application plan for DEMO001.
    {
        "intent": "application_plan",
        "keyword": null,
        "opportunity_type": null,
        "location": null,
        "student_id": null,
        "opportunity_id": "DEMO001"
    }

    User: Check my eligibility for DEMO003.
    {
        "intent": "check_eligibility",
        "keyword": null,
        "opportunity_type": null,
        "location": null,
        "student_id": null,
        "opportunity_id": "DEMO003"
    }

    User: Find suitable internships, check my eligibility, rank them and prepare an application plan.
    {
        "intent": "multi_step_planning",
        "keyword": null,
        "opportunity_type": "Internship",
        "location": null,
        "student_id": null,
        "opportunity_id": null
    }
    """

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0
    )

    ai_output = response.choices[0].message.content

    if not ai_output:
        raise ValueError("The AI returned an empty response.")

    ai_output = ai_output.strip()
    ai_output = ai_output.replace("```json", "")
    ai_output = ai_output.replace("```", "").strip()

    try:
        request_data = json.loads(ai_output)

    except json.JSONDecodeError:
        raise ValueError(
            "The AI returned invalid JSON. Please try again."
        )

    if not isinstance(request_data, dict):
        raise ValueError("The AI response must be a JSON object.")

    # Keep only recognised fields
    allowed_fields = [
        "intent",
        "keyword",
        "opportunity_type",
        "location",
        "student_id",
        "opportunity_id"
    ]

    request_data = {
        field: request_data.get(field)
        for field in allowed_fields
    }

    # Validate intent
    valid_intents = [
        "search_opportunities",
        "match_opportunities",
        "check_eligibility",
        "application_plan",
        "deadline_check",
        "general_question",
        "multi_step_planning"
    ]

    if request_data["intent"] not in valid_intents:
        request_data["intent"] = "general_question"

    # Normalise opportunity type
    opportunity_type = request_data["opportunity_type"]

    if opportunity_type:
        opportunity_type = str(opportunity_type).strip().lower()

        if opportunity_type in ["internship", "internships"]:
            request_data["opportunity_type"] = "Internship"

        elif opportunity_type in ["scholarship", "scholarships"]:
            request_data["opportunity_type"] = "Scholarship"

        else:
            request_data["opportunity_type"] = None

    # The Student ID selected in Streamlit is authoritative.
    # Do not allow the language model or chat message to change it.
    selected_student_id = str(selected_student_id).strip()

    if not re.fullmatch(r"\d{8}", selected_student_id):
        raise ValueError(
            "Invalid Student ID. Student IDs must contain exactly 8 digits."
        )

    request_data["student_id"] = selected_student_id

    # Validate opportunity ID format
    opportunity_id = request_data["opportunity_id"]

    if opportunity_id:
        opportunity_id = str(opportunity_id).strip().upper()

        if re.fullmatch(r"DEMO\d+", opportunity_id):
            request_data["opportunity_id"] = opportunity_id
        else:
            request_data["opportunity_id"] = None

    # Extract opportunity IDs from the original text.
    # Student identity always comes from the Streamlit selection.
    opportunity_ids = re.findall(
        r"\bDEMO\d+\b",
        user_message,
        re.IGNORECASE
    )

    if opportunity_ids:
        request_data["opportunity_id"] = opportunity_ids[0].upper()

    return request_data


def prepare_results_for_ai(data):

    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")

    elif isinstance(data, pd.Series):
        return data.to_dict()

    elif isinstance(data, dict):
        return {
            str(key): prepare_results_for_ai(value)
            for key, value in data.items()
        }

    elif isinstance(data, (list, tuple)):
        return [
            prepare_results_for_ai(item)
            for item in data
        ]

    elif pd.isna(data) if not isinstance(data, (dict, list, tuple, pd.DataFrame, pd.Series)) else False:
        return None

    else:
        return data


# ---------------------------------------------------------------------
# Tool router
# ---------------------------------------------------------------------

def execute_agent_tool(request_data):

    intent = request_data.get("intent")

    student_id = request_data.get("student_id")

    if not student_id or not re.fullmatch(r"\d{8}", str(student_id)):
        return {
            "status": "ERROR",
            "tool": None,
            "message": "A valid 8-digit Student ID is required."
        }

    opportunity_id = request_data.get("opportunity_id")

    keyword = request_data.get("keyword")
    opportunity_type = request_data.get("opportunity_type")
    location = request_data.get("location")

    print("\nAGENT DECISION")
    print("-" * 45)
    print("Intent:", intent)

    def success(tool_name, results):
        return {
            "status": "SUCCESS",
            "tool": tool_name,
            "results": results
        }

    def error(message):
        return {
            "status": "ERROR",
            "tool": None,
            "message": message
        }

    # TOOL 1: SEARCH OPPORTUNITIES

    if intent == "search_opportunities":

        print("Selected tool: search_opportunities()")

        results = search_opportunities(
            keyword=keyword,
            opportunity_type=opportunity_type,
            location=location
        )

        return success(
            "search_opportunities",
            results
        )

    # TOOL 2: MATCH AND RANK OPPORTUNITIES

    elif intent == "match_opportunities":

        print("Selected tool: run_opportunity_agent()")

        student = get_student_profile(student_id)

        if student is None:
            return error(
                f"Student profile {student_id} was not found."
            )

        results = run_opportunity_agent(
            student_id=student_id,
            keyword=keyword,
            opportunity_type=opportunity_type,
            location=location
        )

        # Preserve the original Phase 2 output.
        return success(
            "run_opportunity_agent",
            results
        )

    # TOOL 3: CHECK ELIGIBILITY

    elif intent == "check_eligibility":

        print("Selected tool: check_eligibility()")

        student = get_student_profile(student_id)

        if student is None:
            return error(
                f"Student profile {student_id} was not found."
            )

        if not opportunity_id:
            return {
                "status": "NEEDS_INFORMATION",
                "tool": None,
                "message": "Please provide an opportunity ID."
            }

        opportunity = get_opportunity(opportunity_id)

        if opportunity is None:
            return error(
                f"Opportunity {opportunity_id} was not found."
            )

        results = check_eligibility(
            student,
            opportunity
        )

        return success(
            "check_eligibility",
            results
        )

    # TOOL 4: APPLICATION PLANNING

    elif intent == "application_plan":

        print("Selected tool: create_application_plan()")

        if not opportunity_id:
            return {
                "status": "NEEDS_INFORMATION",
                "tool": None,
                "message": "Please provide an opportunity ID."
            }

        opportunity = get_opportunity(opportunity_id)

        if opportunity is None:
            return error(
                f"Opportunity {opportunity_id} was not found."
            )

        results = create_application_plan(
            opportunity_id=opportunity_id
        )

        return success(
            "create_application_plan",
            results
        )

    # TOOL 5: DEADLINE CHECK

    elif intent == "deadline_check":

        print("Selected tools: search_opportunities() and "
              "rank_opportunities()")

        student = get_student_profile(student_id)

        if student is None:
            return error(
                f"Student profile {student_id} was not found."
            )

        # Retrieve opportunities
        found_opportunities = search_opportunities()

        if found_opportunities.empty:
            return success(
                "deadline_check",
                found_opportunities
            )

        # Rank opportunities using the Phase 2 function
        ranked_results = rank_opportunities(
            student,
            found_opportunities
        )

        # Add deadline priority information
        deadline_results = add_deadline_information(
            ranked_results,
            found_opportunities
        )

        # Show urgent and high-priority deadlines
        if "deadline_priority" in deadline_results.columns:

            deadline_results = deadline_results[
                deadline_results["deadline_priority"].isin(
                    ["URGENT", "HIGH"]
                )
            ]

        return success(
            "deadline_check",
            deadline_results
        )

    # GENERAL QUESTION

    elif intent == "general_question":

        return {
            "status": "GENERAL",
            "tool": None,
            "message": (
                "I can help you find internships and scholarships, "
                "check eligibility, compare opportunities, "
                "review deadlines and prepare application plans."
            )
        }

    # UNKNOWN INTENT

    else:

        return error(
            "I could not identify the requested action."
        )


# ---------------------------------------------------------------------
# Multi-step planning
# ---------------------------------------------------------------------

def create_execution_plan(user_request, student_id):

    request_lower = user_request.lower()

    # Use and validate the Student ID selected in Streamlit.
    student_id = str(student_id).strip()

    if not re.fullmatch(r"\d{8}", student_id):
        raise ValueError("Invalid 8-digit Student ID.")

    plan = []

    # Step 1: Search opportunities
    plan.append({
        "step": 1,
        "tool": "search_opportunities",
        "description": "Find relevant opportunities"
    })

    # Step 2: Assess eligibility
    if any(word in request_lower for word in [
        "eligible", "eligibility", "suitable", "qualify"
    ]):
        plan.append({
            "step": 2,
            "tool": "check_eligibility",
            "description": "Assess student eligibility"
        })

    # Step 3: Rank opportunities
    plan.append({
        "step": len(plan) + 1,
        "tool": "rank_opportunities",
        "description": "Rank opportunities by match score"
    })

    # Step 4: Prepare an application plan
    if any(word in request_lower for word in [
        "application plan",
        "prepare application",
        "application checklist",
        "apply"
    ]):
        plan.append({
            "step": len(plan) + 1,
            "tool": "create_application_plan",
            "description": "Prepare an application checklist"
        })

    return {
        "student_id": student_id,
        "original_request": user_request,
        "steps": plan
    }


def execute_multi_step_plan(plan):

    student_id = plan["student_id"]

    # Retrieve the student's adapted profile
    student = create_adapted_student_profile(student_id)

    # Keep track of execution
    execution_results = []

    # Store intermediate results
    search_results = None
    ranked_results = None
    application_result = None

    print("===== MULTI-STEP AGENT EXECUTION =====")

    for step in plan["steps"]:

        tool_name = step["tool"]

        print(f"\nExecuting Step {step['step']}: {step['description']}")

        try:

            # STEP 1: Search opportunities
            if tool_name == "search_opportunities":

                search_results = search_opportunities(
                    include_demo=True
                )

                output = search_results

            # STEP 2: Check eligibility
            elif tool_name == "check_eligibility":

                if search_results is None or search_results.empty:
                    output = {
                        "message": "No opportunities found."
                    }

                else:
                    eligibility_results = []

                    for _, opportunity in search_results.iterrows():

                        result = check_eligibility(
                            student,
                            opportunity
                        )

                        eligibility_results.append({
                            "opportunity_id": opportunity["opportunity_id"],
                            "title": opportunity["title"],
                            "eligibility": result
                        })

                    output = eligibility_results

            # STEP 3: Rank opportunities
            elif tool_name == "rank_opportunities":

                if search_results is None or search_results.empty:

                    output = pd.DataFrame()

                else:
                    ranked_results = rank_opportunities(
                        student,
                        search_results
                    )

                    output = ranked_results

            # STEP 4: Create an application plan
            elif tool_name == "create_application_plan":

                if ranked_results is None or ranked_results.empty:

                    output = {
                        "message": "No ranked opportunities are available."
                    }

                else:
                    # Prefer the highest-ranked eligible opportunity
                    eligible_results = ranked_results[
                        ranked_results["eligibility"] == "ELIGIBLE"
                    ]

                    if eligible_results.empty:

                        output = {
                            "message": (
                                "No fully eligible opportunity was found. "
                                "An application plan was not automatically created."
                            )
                        }

                    else:
                        best_opportunity_id = (
                            eligible_results.iloc[0]["opportunity_id"]
                        )

                        application_result = create_application_plan(
                            best_opportunity_id
                        )

                        output = application_result

            else:
                output = {
                    "error": f"Unknown tool: {tool_name}"
                }

            execution_results.append({
                "step": step["step"],
                "tool": tool_name,
                "status": "SUCCESS",
                "output": output
            })

            print("Completed successfully.")

        except Exception as error:

            execution_results.append({
                "step": step["step"],
                "tool": tool_name,
                "status": "FAILED",
                "error": str(error)
            })

            print("Execution failed:", error)

    return {
        "student_id": student_id,
        "execution_results": execution_results,
        "search_results": search_results,
        "ranked_results": ranked_results,
        "application_result": application_result
    }


# ---------------------------------------------------------------------
# AI response generation
# ---------------------------------------------------------------------

def generate_agent_response(user_request, tool_result, memory_context=None):

    # Convert tool results into JSON-compatible data
    clean_results = prepare_results_for_ai(tool_result)

    results_json = json.dumps(
        clean_results,
        ensure_ascii=False,
        default=str
    )

    memory_json = json.dumps(
        memory_context or [],
        ensure_ascii=False,
        default=str
    )

    system_prompt = """
    You are a Bhutan Student Internship and Scholarship
    Opportunity Matching Agent.

    Your responsibilities are:
    1. Explain opportunity search results.
    2. Explain eligibility assessments.
    3. Explain application plans and document checklists.
    4. Help students understand upcoming deadlines.
    5. Provide clear and useful responses.

    IMPORTANT RULES:
    - Use the current tool results as the authoritative source
      for opportunity details and eligibility.
    - Never invent opportunities, deadlines, eligibility rules,
      application documents, scores or application links.
    - Previous interactions can help you understand student
      preferences, but must not replace current tool results.
    - If information is unavailable, clearly say so.
    - Opportunities marked DEMO are fictional demonstration
      records, not verified real opportunities.
    - Use simple and understandable English.
    """

    user_prompt = f"""
    STUDENT'S CURRENT REQUEST:
    {user_request}

    CURRENT TOOL RESULTS:
    {results_json}

    PREVIOUS INTERACTIONS FROM MEMORY:
    {memory_json}

    Use previous interactions only when they are relevant
    to understanding the student's preferences or context.

    Answer the student's current request clearly.
    """

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------
# LangGraph state and nodes
# ---------------------------------------------------------------------

class OpportunityAgentState(TypedDict, total=False):

    user_request: str
    student_id: str
    interpreted_request: dict
    tool_result: dict
    selected_tool: str
    status: str
    execution_trace: list
    error: str

    # New field for AI-generated response
    final_response: str


def understand_request_node(state: OpportunityAgentState):

    print("\n[LANGGRAPH] Understanding student request...")

    request = state["user_request"]

    interpreted = understand_student_request(
        request,
        selected_student_id=state["student_id"]
    )

    print("[LANGGRAPH] Intent identified:",
          interpreted["intent"])

    trace = state.get("execution_trace", [])

    trace = trace + ["Request understood"]

    return {
        "interpreted_request": interpreted,
        "execution_trace": trace
    }


def run_tool_node(state, expected_intent):

    print(f"\n[LANGGRAPH] Executing {expected_intent} node...")

    interpreted = state["interpreted_request"]

    # Confirm the correct route was selected
    if interpreted["intent"] != expected_intent:

        return {
            "status": "ERROR",
            "error": "Incorrect LangGraph routing.",
            "selected_tool": None,
            "tool_result": {
                "status": "ERROR",
                "message": "The selected node does not match the intent."
            },
            "execution_trace": (
                state.get("execution_trace", [])
                + ["Routing error"]
            )
        }

    # Execute existing Python tool router
    result = execute_agent_tool(interpreted)

    trace = state.get("execution_trace", [])

    trace = trace + [
        f"Executed {result.get('tool') or expected_intent}"
    ]

    return {
        "tool_result": result,
        "selected_tool": result.get("tool"),
        "status": result.get("status"),
        "execution_trace": trace
    }


def search_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "search_opportunities"
    )


def match_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "match_opportunities"
    )


def eligibility_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "check_eligibility"
    )


def application_plan_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "application_plan"
    )


def deadline_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "deadline_check"
    )


def general_question_node(state: OpportunityAgentState):

    return run_tool_node(
        state,
        "general_question"
    )


def multi_step_planning_node(state):

    print("\nStarting multi-step planning...")

    # Create the execution plan
    plan = create_execution_plan(
        state["user_request"],
        student_id=state["student_id"]
    )

    # Execute the plan
    results = execute_multi_step_plan(plan)

    # Determine execution status
    execution_results = results["execution_results"]

    if execution_results and all(
        item["status"] == "SUCCESS"
        for item in execution_results
    ):
        execution_status = "SUCCESS"

    elif execution_results:
        execution_status = "PARTIAL"

    else:
        execution_status = "FAILED"

    # Prepare results for the AI response generator
    return {
        "selected_tool": "multi_step_planning",
        "status": execution_status,
        "tool_result": results,
        "execution_trace": (
            state.get("execution_trace", [])
            + [
                f"Multi-step plan created with {len(plan['steps'])} steps",
                "Multiple tools executed through the planner"
            ]
        )
    }


def generate_response_node(state: OpportunityAgentState):

    print("\n[LANGGRAPH] Generating AI response...")

    interpreted_request = state.get("interpreted_request", {}) or {}

    student_id = state["student_id"]

    try:
        # Retrieve previous interactions from episodic memory
        try:
            recent_history = get_recent_interactions(
                student_id,
                limit=3
            )
        except Exception:
            recent_history = []

        final_response = generate_agent_response(
            user_request=state["user_request"],
            tool_result=state.get("tool_result", {}),
            memory_context=recent_history
        )

        trace_step = "Retrieved episodic memory and generated AI response"

    except Exception as error:
        final_response = (
            "I was unable to generate a response "
            "from the AI model. The Python tool "
            "execution results are still available. "
            f"Error: {error}"
        )
        trace_step = "AI response generation failed"

    return {
        "final_response": final_response,
        "execution_trace": (
            state.get("execution_trace", []) + [trace_step]
        )
    }


def route_request(state):

    interpreted_request = state.get(
        "interpreted_request", {}
    )

    intent = interpreted_request.get(
        "intent", "general_question"
    )

    # Detect multi-step requests
    if intent == "multi_step_planning":
        return "multi_step_planning"

    # Existing individual routes
    routes = {
        "search_opportunities": "search",
        "match_opportunities": "match",
        "check_eligibility": "eligibility",
        "application_plan": "application_plan",
        "deadline_check": "deadline",
        "general_question": "general"
    }

    return routes.get(intent, "general")


# ---------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------

workflow = StateGraph(OpportunityAgentState)

workflow.add_node("understand_request", understand_request_node)
workflow.add_node("search", search_node)
workflow.add_node("match", match_node)
workflow.add_node("eligibility", eligibility_node)
workflow.add_node("application_plan", application_plan_node)
workflow.add_node("deadline", deadline_node)
workflow.add_node("general", general_question_node)
workflow.add_node("multi_step_planning", multi_step_planning_node)
workflow.add_node("generate_response", generate_response_node)

workflow.add_edge(START, "understand_request")

workflow.add_conditional_edges(
    "understand_request",
    route_request,
    {
        "search": "search",
        "match": "match",
        "eligibility": "eligibility",
        "application_plan": "application_plan",
        "deadline": "deadline",
        "general": "general",
        "multi_step_planning": "multi_step_planning"
    }
)

for node in [
    "search",
    "match",
    "eligibility",
    "application_plan",
    "deadline",
    "general",
    "multi_step_planning"
]:
    workflow.add_edge(node, "generate_response")

workflow.add_edge("generate_response", END)

opportunity_agent_graph = workflow.compile()


# ---------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------

def run_langgraph_agent(user_request, student_id):
    """Run the LangGraph workflow for one selected college Student ID."""

    student_id = str(student_id).strip()

    if not re.fullmatch(r"\d{8}", student_id):
        return {
            "status": "ERROR",
            "error": "Please provide a valid 8-digit Student ID."
        }

    # Confirm that the selected student exists before running the graph.
    try:
        student_profile = get_student_profile(student_id)
    except Exception as error:
        return {
            "status": "ERROR",
            "error": f"Could not load student profile: {error}"
        }

    if student_profile is None:
        return {
            "status": "ERROR",
            "error": f"Student profile {student_id} was not found."
        }

    initial_state = {
        "user_request": user_request,
        "student_id": student_id,
        "execution_trace": []
    }

    print("\n" + "=" * 65)
    print("BHUTAN OPPORTUNITY MATCHING AGENT")
    print("=" * 65)
    print("\nSTUDENT ID:", student_id)
    print("\nSTUDENT REQUEST:")
    print(user_request)

    try:
        final_state = opportunity_agent_graph.invoke(initial_state)

        print("\n" + "=" * 65)
        print("AGENT FINAL RESPONSE")
        print("=" * 65)
        print(final_state.get("final_response"))

        print("\nEXECUTION TRACE:")
        for step in final_state.get("execution_trace", []):
            print("->", step)

        return final_state

    except Exception as error:
        print("\nAGENT EXECUTION FAILED:")
        print(str(error))
        return {
            "status": "ERROR",
            "error": str(error)
        }


def run_agent_with_memory(user_request, student_id):
    """Run the agent and save the interaction for the selected student."""

    student_id = str(student_id).strip()

    final_state = run_langgraph_agent(
        user_request,
        student_id=student_id
    )

    if (
        not isinstance(final_state, dict)
        or (
            final_state.get("status") == "ERROR"
            and "interpreted_request" not in final_state
        )
    ):
        return final_state

    interpreted_request = final_state.get("interpreted_request", {}) or {}
    result_summary = str(final_state.get("final_response") or "")[:1000]

    try:
        final_state["memory_interaction_id"] = save_interaction(
            student_id=student_id,
            user_request=user_request,
            intent=interpreted_request.get("intent", "UNKNOWN"),
            selected_tool=final_state.get("selected_tool", "UNKNOWN"),
            status=final_state.get("status", "UNKNOWN"),
            result_summary=result_summary
        )
    except Exception as error:
        # Do not fail the assistant response if memory storage is unavailable.
        print("Could not save interaction:", error)

    return final_state


def ask_agent(user_request, student_id):
    """Public function called by the Streamlit interface."""

    return run_agent_with_memory(
        user_request,
        student_id=student_id
    )
