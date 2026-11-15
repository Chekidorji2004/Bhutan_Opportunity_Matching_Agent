# Bhutan Student Internship and Scholarship Opportunity Matching Agent

An AI-powered opportunity matching agent designed to help Bhutanese students discover internships and scholarships, evaluate their eligibility, identify suitable opportunities, and prepare application plans based on their academic profiles, skills, interests, and career goals.

The system uses **LangGraph, an LLM through OpenRouter, Python-based opportunity matching tools, SQLite episodic memory, and Streamlit** to provide an interactive AI-assisted opportunity discovery experience.

---

## 1. Project Overview

Finding suitable internships and scholarships can be challenging for students because opportunity information is scattered across different platforms, eligibility requirements vary, and application deadlines can be difficult to track.

This project addresses these challenges by developing an AI agent that assists students throughout the opportunity discovery and application-planning process.

### Main Objectives

- Help students discover relevant internships and scholarships.
- Match opportunities with students' academic backgrounds, skills, interests, and preferences.
- Evaluate eligibility based on opportunity requirements.
- Rank opportunities according to eligibility and matching scores.
- Generate personalised application plans and document checklists.
- Track application readiness and opportunity deadlines.
- Maintain student-specific interaction history and preferences.

## 2. Key Features

### AI Opportunity Assistant
- Accepts natural-language questions from students.
- Uses an LLM to interpret requests and coordinate relevant tools.
- Provides opportunity recommendations and application guidance.

### Student Profile Management
- Supports multiple student profiles.
- Retrieves profiles using student IDs.
- Uses academic qualifications, GPA, skills, interests, and career goals for matching.

### Opportunity Search
- Searches internship and scholarship opportunities.
- Supports filtering based on opportunity type, location, and relevant criteria.

### Eligibility Assessment
- Evaluates citizenship, education level, field of study, and GPA requirements.
- Classifies students as:
  - ELIGIBLE
  - NOT ELIGIBLE
  - NEEDS VERIFICATION

### Opportunity Matching and Ranking
Uses a weighted matching formula:

\[
S = 0.30F + 0.25K + 0.20I + 0.15T + 0.10L
\]

Where:

- F = Field of study similarity (30%)
- K = Skills similarity (25%)
- I = Interests similarity (20%)
- T = Preferred opportunity type (15%)
- L = Location preference (10%)

The matching score represents profile similarity, not the probability of being selected.

### Application Planning
- Generates application plans.
- Provides document checklists.
- Calculates application readiness based on completed checklist items.

### Deadline Tracking
Classifies opportunities according to their deadlines:

- EXPIRED
- URGENT
- HIGH
- MEDIUM
- LOW
- UNKNOWN

### Episodic Memory
- Stores student-specific interactions and preferences using SQLite.
- Maintains separate interaction histories for different student IDs.
- Supports reviewing previous interactions.

### Feedback Mechanism
Allows students to provide feedback on their interactions with the agent.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangGraph | Agent workflow orchestration |
| OpenRouter | LLM API access |
| OpenAI Python Client | Communication with the OpenRouter API |
| Streamlit | Interactive web application |
| Pandas | Dataset processing and manipulation |
| SQLite | Student-specific episodic memory |
| CSV | Student profiles and opportunity datasets |
| Git and GitHub | Version control and source-code management |
| Streamlit Community Cloud | Application deployment |

---

## 4. Project Structure

```text
Bhutan_Opportunity_Matching_Agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── agent_memory.db
│   ├── opportunities.csv
│   └── student_profiles.csv
│
├── src/
│   ├── tools/
│   │   └── opportunity_tools.py
│   │
│   ├── agent/
│   │   └── agent_workflow.py
│   │
│   └── memory/
│       └── episodic_memory.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── test_cases.csv
│   └── results/
│
├── main.py
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── Bhutan_Opportunity_Agent.ipynb
└── README.md
```

---

## 5. Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

Make sure you have installed:

- Python 3.10 or later (use a version compatible with the packages in `requirements.txt`).
- Git.
- A code editor such as Visual Studio Code.
- An OpenRouter API key.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chekidorji2004/Bhutan_Opportunity_Matching_Agent.git
```

Navigate into the project directory:

```bash
cd Bhutan_Opportunity_Matching_Agent
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the API Key

Create a `.env` file in the project root directory.

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Replace `your_openrouter_api_key` with your actual API key.

**Important:** Never upload your `.env` file or expose your API key in the GitHub repository. The `.gitignore` file should exclude `.env`.

### Step 5: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 6. Application Pages

The application contains five main pages:

| Page | Description |
|---|---|
| Home | Introduces the system and its main functionalities. |
| AI Assistant | Allows students to ask questions about internships, scholarships, eligibility, and application planning. |
| My Profile | Displays the selected student's academic and personal opportunity preferences. |
| My Application Plans | Displays application plans and document checklist progress. |
| Interaction History | Displays previously stored interactions and student-specific history. |

---

## 7. Dataset Description

The project uses two CSV datasets.

### Student Profiles

File: `data/student_profiles.csv`

Contains demonstration student profiles with information such as:

- Student ID
- Citizenship
- Education level
- Degree and field of study
- Year of study
- GPA
- Skills
- Interests
- Preferred opportunity type
- Preferred location
- Preferred mode
- Career goal

The current dataset contains three fictional student profiles for demonstration and testing.

### Opportunity Dataset

File: `data/opportunities.csv`

Contains demonstration internship and scholarship opportunities, including:

- Opportunity ID
- Opportunity title
- Opportunity type
- Description
- Eligibility requirements
- Required skills
- Location
- Deadline
- Other relevant opportunity information

**Dataset Disclaimer:** The current opportunity records are fictional demonstration data. They are intended for testing and demonstrating the agent's functionality and should not be treated as verified or currently available opportunities.

---

## 8. Evaluation and Testing

The project includes an offline functional evaluation framework.

Evaluation files are located in:

```text
evaluation/
├── evaluate.py
├── test_cases.csv
└── results/
```

### Running the Evaluation

From the project root directory, execute:

```bash
python -m evaluation.evaluate
```

### Evaluation Results

The system was evaluated using 12 predefined offline functional test cases.

| Evaluation Metric | Result |
|---|---:|
| Total test cases | 12 |
| Passed test cases | 12 |
| Failed test cases | 0 |
| Pass rate | 100% |

The evaluation covered:

- Student profile retrieval.
- Internship and scholarship searches.
- Eligibility assessment.
- Opportunity ranking.
- Application plan generation.
- Application readiness calculation.
- Deadline classification.
- Student-specific memory isolation.

All 12 predefined offline functional tests passed.

**Evaluation Note:** These results demonstrate the successful execution of the predefined functional tests. They do not establish overall real-world recommendation accuracy, LLM response quality, or production readiness.

---

## 9. Safety, Privacy and Ethical Considerations

The project considers several important safety and ethical aspects:

- Uses fictional student profiles for demonstration and testing.
- Keeps API keys outside the public source-code repository.
- Separates stored interaction histories by student ID.
- Uses rule-based eligibility assessment for structured requirements.
- Distinguishes matching scores from actual selection probabilities.
- Does not guarantee internship or scholarship selection.
- Identifies demonstration opportunity records as unverified.

**Privacy Limitation:** Student ID selection is used to retrieve a profile; it is not a secure authentication mechanism. The demonstration application should not be used to store sensitive real student information without appropriate authentication and data-protection measures.

---

## 10. Limitations

The current implementation has the following limitations:

1. The opportunity dataset contains fictional demonstration records rather than verified live opportunities.
2. The system does not automatically retrieve real-time opportunities from external websites.
3. Matching scores depend on predefined weights and available profile information.
4. Eligibility assessment is limited to the requirements represented in the dataset.
5. LLM-generated responses may occasionally contain inaccurate or incomplete information.
6. SQLite local-file persistence may not be reliable across cloud restarts or redeployments.
7. Student ID selection does not provide secure user authentication.
8. The current feedback mechanism does not automatically retrain or update the LLM.

---

## 11. Future Improvements

Potential improvements include:

- Integrating verified internship and scholarship sources in Bhutan and internationally.
- Developing a more comprehensive opportunity database.
- Implementing secure student registration and authentication.
- Integrating a persistent cloud database.
- Improving matching through advanced recommendation techniques.
- Adding document upload and application assistance.
- Introducing automated deadline notifications.
- Expanding the evaluation framework with more student profiles and real-world scenarios.
- Improving safety checks and evaluating LLM responses systematically.

---

## 12. Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:** https://bhutanopportunitymatchingagent-fyjpwbzgnfjyduegtxyai3.streamlit.app/

**GitHub Repository:** [Bhutan Opportunity Matching Agent](https://github.com/Chekidorji2004/Bhutan_Opportunity_Matching_Agent)

The deployed application uses Streamlit Cloud Secrets to securely configure the OpenRouter API key.

---

## 13. References and Acknowledgements

The project uses the following open-source technologies and documentation:

- [Python Documentation](https://docs.python.org/3/)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

Additional academic references and AI-assistance attribution are provided in the technical report.

---

## 14. Author

**Project:** Bhutan Student Internship and Scholarship Opportunity Matching Agent

**Course:** DAT404 – AI Agent

**Institution:** Sherubtse College

**Project Type:** Individual AI Agent Project

---

*This project was developed for academic purposes to demonstrate the design and implementation of an LLM-powered AI agent for student internship and scholarship opportunity discovery, matching, eligibility assessment, and application planning.*
