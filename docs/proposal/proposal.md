# School of Computing &mdash; Year 4 Project Proposal Form


## SECTION A

|                     |                   |
|---------------------|-------------------|
|Project Title:       | ContractGuard     |
|Student 1 Name:      | Chunyang Wang     |
|Student 1 ID:        | 21115214          |
|Student 2 Name:      | Justin Siak       |
|Student 2 ID:        | 22449184          |
|Project Supervisor:  | Graham Healy      |


## SECTION B


### Introduction


Modern software systems rely heavily on APIs to connect distributed services. Ensuring these APIs behave as defined in their specifications is essential for reliability and maintainability. However, in enterprise level or big project manual testing is very time-consuming and energy-consuming.

This project proposes an intelligent platform that automates API contract verification and test generation. It automatic parseing uploaded or pushed to Github API specifications, applies rule-based logic defined ourselves to generate tests, and validates API behaviour automatically. For complex cases, an integrated LLM assists by generating additional tests and producing human-readable explanations for failed cases.

A web-based dashboard allows users to upload API specifications, view test outcomes, explore insights from the LLM, and compare API quality. The system also supports automated testing pipelines using GitHub Actions, enabling continuous verification whenever new API specifications are pushed or updated.

By combining automated testing with AI-driven interpretability, the platform aims to make API validation more efficient.


### Outline


The proposed project will develop an automated platform for API contract verification and intelligent test generation.  
It consists of three main components:

- **Backend Service (A):** Handles API specification ingestion, automatic api specifications sparsing system, rule-based test case generation, and execution.  
- **Frontend Dashboard (B):** Frontend interface, visualizing test results and history, LLM integration. 
pre-baked prompts, ci/cd generation, jenkins pipeline, Github action.
- **LLM Integration (C):** Enhances test generation and result interpretation through AI-driven explanations, api scoring system,  LLM exposed through frontend UI.


### Background

From Chunyang’s internship experience, issues often arose during cluster patching and system updates due to failing APIs within container images. Similarly, when developing the MCP server integrated with LLMs, a single API failure could cause the entire system to malfunction. These situations highlight the importance of continuously verifying and maintaining API reliability.

Justin also observed that one of his colleagues used Kafka-based scripts for API monitoring during his internship, but the approach was error-prone and not user-friendly.

After researching existing tools and platforms, we found that most API testing solutions require significant manual setup and maintenance. They also lack advanced capabilities such as AI-driven explaintion, test generation system, interpretability, and open-source accessibility. These gaps motivated us to design and build our own automated platform.

### Achievements

The project will deliver a platform that automatically verifies API contracts, generates test cases, and provides intelligent feedback using LLMs.  
Key functions include:

- **Automated API Specification Parsing:** Supports both OpenAPI and Swagger formats for extracting endpoint details and parameters.  
- **Rule-Based and AI-Assisted Test Generation:** Creates functional and edge test cases, with LLM support for complex or ambiguous schemas.  
- **Automated Test Execution:** Runs tests efficiently using pytest or Postman CLI and records outcomes.  
- **Interactive Dashboard:** Allows users to upload API specifications, view test results, and interpret with LLM, can also see scoring system for API quality.
- **Continuous Verification Pipeline:** Integrates with GitHub Actions for automatic testing when specifications are updated.

The platform is designed primarily for **developers**, **QA engineers**, and **Computer Scinece Students** who want to ensure the reliability and consistency of their APIs without extensive manual setup.


### Justification

This project provides a practical and automated solution for developers to continuously verify API correctness with minimal effort. By combining rule-based logic with LLM-driven interpretation, it not only detects failures but also explains them in a human-readable way, helping developers quickly understand and fix issues.

At the same time, there are many possibilities for interaction with LLM, while the integration with automated pipelines ensures that verification becomes a natural part of the development process. Overall, the system promotes higher API quality, easier maintenance, and faster debugging across real-world software environments.


### Programming language(s)

The project will primarily use the following languages:

- **Python** – for backend development using FastAPI, rule-based test generation, and LLM integration.  
- **JavaScript / TypeScript** – for frontend development with React and Node.js for Swagger parsing support.  
- **SQL** – for storing API specifications, test cases, results, and user interaction data.   
- **YAML / JSON** – for representing and parsing API specifications (OpenAPI, Swagger).  
- **Shell / YAML (GitHub Actions)** – for defining CI/CD automation pipelines.

**Optional or supplementary languages:**
- **Bash** – for Docker and testing scripts.  
- **Markdown** – for documentation and reporting.

### Programming tools / Tech stack

The system will incorporate a modern and modular tech stack:

- **Backend Framework:** FastAPI (Python).  
- **Frontend Framework:** React with Tailwind CSS.  
- **API Parsing:** OpenAPI (Python) and Swagger (Node.js) parsers for flexible spec ingestion.  
- **Test Runner:** pytest + requests or Postman CLI  
- **Containerization:** Docker 
- **Version Control:** Git and GitHub
- **CI/CD:** GitHub Actions (and optionally Jenkins). 
- **LLM Integration:** OpenAI API, Ollama, or local models for AI-driven test generation and explanation.

### Hardware


Currently no expectation for any non-standard hardware to be required.

### Learning Challenges


- Parsing and interpreting API specifications, especially complex schemas.
- Integrating with CI/CD workflow.
- Using LLMs for natural-language explanations.
- Database design and result persistence.
- Security and sandboxing considerations.

### Breakdown of work


|Component|Chunyang|Justin|
|---------|---------|------|
|Requirements & Design|System Architecture Design; Research: Test Generation Techniques and API spec standards|System Architecture Design; Research: UI needs, reporting, and CI/CD workflows|
|API Spec Parsing & Handling|Implement spec upload endpoints in FastAPI; Build the parser to extract endpoints, schemas, constraints; Handle validation and error reporting.|Create frontend UI to upload API specs and view parsing results; Design user flows for how specs are submitted and processed.|
|Test Case Generation Engine| Develop rule-based engine for generating test cases from schema consrtraints; Implement fallback LLM integration for complex schemas and enhancements.|Work on visualisation of generated tests in the dashboard; Help design mapping logic for test generation from schema.|
|Test Runner & Execution|Build the test execution layer; Implement orchestration logic to trigger tests and collect results; Design structured result formats.|Integrate test execution trigger into frontend; Handle real-time or asynchronous result updates in UI.|
|Data Layer & Storage| Design PostgreSQL schema; Implement database queries and result persistence| Integrate frontend with backend data endpoints|
|CI/CD Integration| Provide backend endpoints and CLI entry points for CI/CD triggers|Implement pipeline to run verification on push; Handle automation scripts, container builds, and reporting integration.|
|Frontend & Dashboard Development| Assist with result APIs, provide backend data for charts| Develop full React dashboard with tailwind styling; Implement result views and feedback.|
|LLM Integration| Build backend integration with OpenAI (or local model); Process raw text test data into input format for LLM| Build LLM sandbox operability; Design presentation for LLM querying and feedback.|
|Containerisation and Deployment| Write Dockerfiles for backend and test runner; Test orchestration inside Docker.| Compose frontend, backend, and DB with Docker Compose; Ensure CI/CD uses the containerised setup.|
|Testing & Validation|Develop internal test APIs for validation; Write backend unit/integration tests| Perform UI testing, pipeline dry runs, and demo preparation.| 

### Risk Register


| Description | Likelyhood | Severity | Mitigation |
|-------------|------------|----------|------------|
|Complexity of API specifications is higher than expected, making parsing and test generation difficult.             |Medium            |High          |Start with OpenAPI v3 support only and test early with real-world specs. Build a fallback LLM-based generator for complex schemas.            |
|LLM integration underperforms (e.g., produces inaccurate or irrelevant explanations).             |Medium            |Medium          |Treat LLM as an enhancement, not a core dependency. Ensure the system remains functional without it. Use prompt engineering and limit output scope.            |
|CI/CD automation fails to trigger correctly (e.g, GitHub Actions not running or misconfigured).             |Low            |Medium          |Manually test CI/CD pipeline in early sprints. Document setup clearly and keep a manual trigger option in the UI.            |
|Third-party API libraries or LLM API become unavailable (e.g rate-limited, pricing changes, or outages).|Medium|Medium|Cache responses where possible, abstract LLM usage behind an interface, and ensure the system works without external APIs.|
|Generated tests do not reflect real-world scenarios accurately (false positives/negatives).|Medium|High|Validate generated tests against known APIs with expected results. Manually review generated test logic during development.|

#### Student 1

> *Student 1 should complete this section.*

#### Student 2

> *Student 2 should complete this section.*

