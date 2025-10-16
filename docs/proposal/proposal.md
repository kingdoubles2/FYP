# School of Computing &mdash; Year 4 Project Proposal Form

> Edit (then commit and push) this document to complete your proposal form.
> Make use of figures / diagrams where appropriate.
>
> Do not rename this file.

## SECTION A

|                     |                   |
|---------------------|-------------------|
|Project Title:       | VerAPI Contract Grard            |
|Student 1 Name:      | Chunyang Wang     |
|Student 1 ID:        | 21115214          |
|Student 2 Name:      | xxxxxx            |
|Student 2 ID:        | xxxxxx            |
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

> Describe any non-standard hardware components which will be required.

### Learning Challenges

> List the main new things (technologies, languages, tools, etc) that you will have to learn.

### Breakdown of work

> Clearly identify who will undertake which parts of the project.
>
> It must be clear from the explanation of this breakdown of work both that each student is responsible for
> separate, clearly-defined tasks, and that those responsibilities substantially cover all of the work required
> for the project.


### Risk Register

> You are to complete a risk register to demonstrate how you will account for the potential risks (i.e. something going wrong) when youa re working on your proiject. 
> The purpose of a risk register is to identify potential issues beforehand and plan mitigation strategies to ensure the success of the project. 
> The reisks are different for each project by some examples include 

> if you are using physical components (e..g sensors), what happens if they never arrive, or if they break, or if they are not suitable?
> if you are creating an AI/ML method , what happens if there is no suitable training data, or if the model performs poorly after training? 

| Description | Likelyhood | Severity | Mitigation |
|-------------|------------|----------|------------|
|             |            |          |            |
|             |            |          |            |
|             |            |          |            |

#### Student 1

> *Student 1 should complete this section.*

#### Student 2

> *Student 2 should complete this section.*

## Example

> Example: Here's how you can include images in markdown documents...

<!-- Basically, just use HTML! -->

<p align="center">
  <img src="./res/cat.png" width="300px">
</p>

