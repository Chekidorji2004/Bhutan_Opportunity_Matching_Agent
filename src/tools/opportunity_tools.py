"""
Opportunity tools for the Bhutan Opportunity Matching Agent.

Functions were moved from Bhutan_Opportunity_Agent.ipynb.
Only the file paths (and one import location) were changed.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from src.memory.episodic_memory import get_student_preferences


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPPORTUNITIES_PATH = PROJECT_ROOT / "data" / "opportunities.csv"
STUDENTS_PATH = PROJECT_ROOT / "data" / "student_profiles.csv"

# .fillna("") matches the notebook; the eligibility checker relies on it
opportunities = pd.read_csv(OPPORTUNITIES_PATH, dtype={"opportunity_id": str}).fillna("")
# Student IDs must be strings so IDs such as 07230000 retain their leading zero.
students = pd.read_csv(STUDENTS_PATH, dtype={"student_id": str}).fillna("")


def normalize_student_id(student_id):
    """Return a validated 8-digit Student ID as a string."""
    if student_id is None:
        return None

    normalized_id = str(student_id).strip()

    if not normalized_id.isdigit() or len(normalized_id) != 8:
        return None

    return normalized_id

def split_values(value):

    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        item.strip().lower()
        for item in str(value).split(";")
        if item.strip()
    ]


def search_opportunities(
    keyword=None,
    opportunity_type=None,
    location=None,
    include_demo=True
):
    """
    Search opportunities using multiple criteria.

    Parameters:
        keyword: Search terms.
        opportunity_type: Internship or Scholarship.
        location: Preferred location.
        include_demo: Whether demonstration records are included.

    Returns:
        DataFrame containing matching opportunities.
    """

    results = opportunities.copy()

    # Exclude demonstration data when requested
    if not include_demo:
        results = results[
            results["is_demo"].astype(str).str.lower() != "true"
        ]

    # Keyword search
    if keyword:

        search_columns = [
            "title",
            "description",
            "eligible_fields",
            "required_skills",
            "benefits"
        ]

        search_text = results[search_columns].astype(str).agg(
            " ".join, axis=1
        ).str.lower()

        keywords = str(keyword).lower().split()

        mask = pd.Series(True, index=results.index)

        for word in keywords:
            mask &= search_text.str.contains(
                word, regex=False, na=False
            )

        results = results[mask]

    # Opportunity type
    if opportunity_type:
        results = results[
            results["opportunity_type"].str.lower()
            == str(opportunity_type).lower()
        ]

    # Location
    if location:
        location = str(location).lower()

        results = results[
            results["location"].apply(
                lambda value: location in str(value).lower()
            )
        ]

    return results.reset_index(drop=True)


def check_eligibility(student, opportunity):

    reasons = []
    failed_requirements = []
    missing_information = []

    # 1. Citizenship
    accepted_citizenships = split_values(
        opportunity["eligible_citizenship"]
    )

    student_citizenship = str(
        student["citizenship"]
    ).strip().lower()

    if accepted_citizenships:

        if not student_citizenship:
            missing_information.append("Citizenship")

        elif student_citizenship in accepted_citizenships:
            reasons.append("Citizenship requirement satisfied")

        else:
            failed_requirements.append(
                "Citizenship requirement not met"
            )

    # 2. Education level
    accepted_education = split_values(
        opportunity["eligible_education_levels"]
    )

    student_education = str(
        student["education_level"]
    ).strip().lower()

    if accepted_education:

        if not student_education:
            missing_information.append("Education level")

        elif student_education in accepted_education:
            reasons.append("Education level requirement satisfied")

        else:
            failed_requirements.append(
                "Education level requirement not met"
            )

    # 3. Field of study
    eligible_fields = split_values(
        opportunity["eligible_fields"]
    )

    student_field = str(
        student["field_of_study"]
    ).strip().lower()

    if eligible_fields:

        if not student_field:
            missing_information.append("Field of study")

        else:
            field_match = any(
                field in student_field
                or student_field in field
                for field in eligible_fields
            )

            if field_match:
                reasons.append("Field of study requirement satisfied")
            else:
                failed_requirements.append(
                    "Field of study does not match"
                )

    # 4. Minimum GPA
    minimum_gpa = opportunity["min_gpa"]

    if str(minimum_gpa).strip():

        try:
            required_gpa = float(minimum_gpa)

            if str(student["gpa"]).strip() == "":
                missing_information.append("GPA")

            else:
                student_gpa = float(student["gpa"])

                if student_gpa >= required_gpa:
                    reasons.append("Minimum GPA requirement satisfied")
                else:
                    failed_requirements.append(
                        f"GPA below minimum of {required_gpa}"
                    )

        except (ValueError, TypeError):
            missing_information.append("Valid GPA information")

    # 5. Required skills
    required_skills = split_values(
        opportunity["required_skills"]
    )

    student_skills = split_values(
        student["skills"]
    )

    matched_skills = [
        skill for skill in required_skills
        if skill in student_skills
    ]

    skill_gaps = [
        skill for skill in required_skills
        if skill not in student_skills
    ]

    # Final eligibility decision
    if failed_requirements:
        status = "NOT ELIGIBLE"

    elif missing_information:
        status = "NEEDS VERIFICATION"

    else:
        status = "ELIGIBLE"

    return {
        "eligibility_status": status,
        "reasons": reasons,
        "failed_requirements": failed_requirements,
        "missing_information": missing_information,
        "matched_skills": matched_skills,
        "skill_gaps": skill_gaps
    }


def calculate_match_score(student, opportunity):

    score = 0
    reasons = []

    # 1. Field of study: 30 points
    student_field = str(
        student["field_of_study"]
    ).lower()

    eligible_fields = split_values(
        opportunity["eligible_fields"]
    )

    field_match = any(
        field in student_field or student_field in field
        for field in eligible_fields
    )

    if field_match:
        score += 30
        reasons.append("Field of study matches")

    # 2. Skills: 25 points
    student_skills = split_values(student["skills"])
    required_skills = split_values(opportunity["required_skills"])

    matched_skills = [
        skill for skill in required_skills
        if skill in student_skills
    ]

    if required_skills:
        skill_score = (
            len(matched_skills) / len(required_skills)
        ) * 25

        score += skill_score

        if matched_skills:
            reasons.append(
                "Matching skills: " + ", ".join(matched_skills)
            )

    # 3. Interests: 20 points
    student_interests = split_values(student["interests"])

    opportunity_text = " ".join([
        str(opportunity["title"]),
        str(opportunity["description"]),
        str(opportunity["eligible_fields"]),
        str(opportunity["required_skills"]),
        str(opportunity["benefits"])
    ]).lower()

    matched_interests = [
        interest for interest in student_interests
        if interest in opportunity_text
    ]

    if student_interests:
        interest_score = (
            len(matched_interests) / len(student_interests)
        ) * 20

        score += interest_score

        if matched_interests:
            reasons.append(
                "Matching interests: " + ", ".join(matched_interests)
            )

    # 4. Preferred opportunity type: 15 points
    preferred_type = str(
        student["preferred_opportunity_type"]
    ).lower()

    actual_type = str(
        opportunity["opportunity_type"]
    ).lower()

    if preferred_type == actual_type:
        score += 15
        reasons.append("Preferred opportunity type matches")

    # 5. Preferred location: 10 points
    preferred_locations = split_values(
        student["preferred_location"]
    )

    opportunity_location = str(
        opportunity["location"]
    ).lower()

    if any(
        location in opportunity_location
        for location in preferred_locations
    ):
        score += 10
        reasons.append("Preferred location matches")

    return round(score, 2), reasons


def rank_opportunities(student, opportunity_data):

    records = []

    for _, opportunity in opportunity_data.iterrows():

        eligibility = check_eligibility(student, opportunity)

        score, reasons = calculate_match_score(
            student,
            opportunity
        )

        records.append({
            "opportunity_id": opportunity["opportunity_id"],
            "title": opportunity["title"],
            "opportunity_type": opportunity["opportunity_type"],
            "eligibility": eligibility["eligibility_status"],
            "match_score": score,
            "match_reasons": "; ".join(reasons),
            "skill_gaps": ", ".join(eligibility["skill_gaps"]),
            "failed_requirements": "; ".join(
                eligibility["failed_requirements"]
            ),
            "missing_information": ", ".join(
                eligibility["missing_information"]
            )
        })

    ranked_df = pd.DataFrame(records)

    priority = {
        "ELIGIBLE": 0,
        "NEEDS VERIFICATION": 1,
        "NOT ELIGIBLE": 2
    }

    ranked_df["priority"] = ranked_df["eligibility"].map(priority)

    ranked_df = ranked_df.sort_values(
        by=["priority", "match_score"],
        ascending=[True, False]
    )

    return ranked_df.drop(
        columns=["priority"]
    ).reset_index(drop=True)


def generate_application_checklist(
    opportunity,
    available_documents=None
):
    """
    Generate an application checklist.

    available_documents represents documents the student
    confirms they already have.
    """

    required_documents = split_values(
        opportunity["required_documents"]
    )

    available_documents = [
        item.lower().strip()
        for item in (available_documents or [])
    ]

    checklist = []

    for document in required_documents:

        if document in available_documents:
            status = "COMPLETED"
        else:
            status = "PENDING"

        checklist.append({
            "document": document.title(),
            "status": status
        })

    return pd.DataFrame(checklist)


def calculate_application_readiness(checklist):

    if checklist.empty:
        return 100

    completed = (
        checklist["status"] == "COMPLETED"
    ).sum()

    total = len(checklist)

    return round((completed / total) * 100, 2)


def calculate_deadline_priority(deadline, reference_date=None):

    if reference_date is None:
        reference_date = date.today()

    deadline_date = pd.to_datetime(
        deadline,
        errors="coerce"
    )

    if pd.isna(deadline_date):
        return {
            "days_remaining": None,
            "priority": "UNKNOWN"
        }

    deadline_date = deadline_date.date()

    days_remaining = (deadline_date - reference_date).days

    if days_remaining < 0:
        priority = "EXPIRED"

    elif days_remaining <= 7:
        priority = "URGENT"

    elif days_remaining <= 30:
        priority = "HIGH"

    elif days_remaining <= 60:
        priority = "MEDIUM"

    else:
        priority = "LOW"

    return {
        "days_remaining": days_remaining,
        "priority": priority
    }


def add_deadline_information(
    ranked_df,
    opportunity_data,
    reference_date=None
):

    result = ranked_df.copy()

    deadline_map = opportunity_data.set_index(
        "opportunity_id"
    )["deadline"]

    result["deadline"] = result["opportunity_id"].map(
        deadline_map
    )

    deadline_results = result["deadline"].apply(
        lambda value: calculate_deadline_priority(
            value,
            reference_date
        )
    )

    result["days_remaining"] = deadline_results.apply(
        lambda value: value["days_remaining"]
    )

    result["deadline_priority"] = deadline_results.apply(
        lambda value: value["priority"]
    )

    return result


def run_opportunity_agent(
    student_id,
    keyword=None,
    opportunity_type=None,
    location=None,
    include_demo=True
):
    """
    Coordinate the opportunity matching tools.

    Returns:
        Dictionary containing student information,
        search results and ranked recommendations.
    """

    # STEP 1: Validate and retrieve the selected student's profile.
    normalized_student_id = normalize_student_id(student_id)

    if normalized_student_id is None:
        return {
            "success": False,
            "message": "Invalid Student ID. Please provide an 8-digit ID."
        }

    student = get_student_profile(normalized_student_id)

    if student is None:
        return {
            "success": False,
            "student_id": normalized_student_id,
            "message": f"Student ID {normalized_student_id} was not found."
        }

    # STEP 2: Search opportunities
    search_results = search_opportunities(
        keyword=keyword,
        opportunity_type=opportunity_type,
        location=location,
        include_demo=include_demo
    )

    if search_results.empty:
        return {
            "success": True,
            "student_id": normalized_student_id,
            "student_name": student["name"],
            "message": "No matching opportunities were found.",
            "recommendations": pd.DataFrame()
        }

    # STEP 3: Eligibility checking and ranking
    ranked_results = rank_opportunities(
        student,
        search_results
    )

    # STEP 4: Add deadline information
    final_results = add_deadline_information(
        ranked_results,
        search_results
    )

    # STEP 5: Prepare final recommendations
    return {
        "success": True,
        "student_id": normalized_student_id,
        "student_name": student["name"],
        "message": "Opportunity matching completed successfully.",
        "recommendations": final_results
    }


def create_application_plan(
    opportunity_id,
    available_documents=None,
    student_id=None
):
    """
    Generate an application preparation plan for a selected
    opportunity.
    """

    opportunity_id = str(opportunity_id).strip()

    matches = opportunities[
        opportunities["opportunity_id"].astype(str).str.strip().str.upper()
        == opportunity_id.upper()
    ]

    if matches.empty:
        return {
            "success": False,
            "student_id": normalize_student_id(student_id),
            "message": "Opportunity not found."
        }

    normalized_student_id = None

    if student_id is not None:
        normalized_student_id = normalize_student_id(student_id)

        if normalized_student_id is None:
            return {
                "success": False,
                "message": "Invalid Student ID. Please provide an 8-digit ID."
            }

        if get_student_profile(normalized_student_id) is None:
            return {
                "success": False,
                "student_id": normalized_student_id,
                "message": f"Student ID {normalized_student_id} was not found."
            }

    opportunity = matches.iloc[0]

    checklist = generate_application_checklist(
        opportunity,
        available_documents
    )

    readiness = calculate_application_readiness(checklist)

    return {
        "success": True,
        "student_id": normalized_student_id,
        "opportunity_id": opportunity["opportunity_id"],
        "opportunity": opportunity["title"],
        "checklist": checklist,
        "readiness": readiness,
        "application_url": opportunity["application_url"],
        "source_url": opportunity["source_url"]
    }


def get_student_profile(student_id):
    """Retrieve one student profile using an exact 8-digit ID match."""
    normalized_student_id = normalize_student_id(student_id)

    if normalized_student_id is None:
        return None

    student_record = students[
        students["student_id"].astype(str).str.strip()
        == normalized_student_id
    ]

    if student_record.empty:
        return None

    return student_record.iloc[0]


def get_opportunity(opportunity_id):
    """Retrieve an opportunity by its ID, ignoring letter case."""
    if opportunity_id is None:
        return None

    normalized_opportunity_id = str(opportunity_id).strip().upper()

    opportunity_record = opportunities[
        opportunities["opportunity_id"].astype(str).str.strip().str.upper()
        == normalized_opportunity_id
    ]

    if opportunity_record.empty:
        return None

    return opportunity_record.iloc[0]


def create_adapted_student_profile(student_id):
    """Create a copy of a student's profile with only their learned preferences."""
    normalized_student_id = normalize_student_id(student_id)

    if normalized_student_id is None:
        return None

    student = get_student_profile(normalized_student_id)

    if student is None:
        return None

    student = student.copy()

    # Preferences are retrieved separately for this exact Student ID.
    preferences = get_student_preferences(normalized_student_id) or {}

    # Start with existing profile values
    updated_interests = split_values(
        student.get("interests", "")
    )

    updated_locations = split_values(
        student.get("preferred_location", "")
    )

    # Add learned interests without duplicates
    for interest in preferences.get("Interest", []):

        if interest.lower() not in updated_interests:
            updated_interests.append(interest.lower())

    # Add learned locations without duplicates
    for location in preferences.get("Location", []):

        if location.lower() not in updated_locations:
            updated_locations.append(location.lower())

    # Apply learned opportunity type if available
    learned_types = preferences.get("Opportunity type", [])

    if learned_types:
        student["preferred_opportunity_type"] = learned_types[-1]

    student["interests"] = "; ".join(updated_interests)
    student["preferred_location"] = "; ".join(updated_locations)

    return student