# ContractGuard

ContractGuard is an API contract testing platform that checks whether an API behaves according to its specification. It reads API specifications, generates contract based test cases, runs those tests against real endpoints, and presents the results through a simple web interface.

The project is designed to make API contract verification easier, more repeatable, and more understandable for developers.

---

## Features

- Parse API specifications such as OpenAPI JSON/YAML files.
- Convert API specifications into a structured internal representation.
- Generate deterministic contract based API test cases.
- Execute generated tests against live API endpoints.
- Display generated tests and execution results through a React frontend.
- Support manual review and editing of generated test cases.
- Record test outcomes such as passed, failed, and skipped.
- Use LLM support to explain complex API test failures.
- Suggest follow-up or repair test cases based on failed results.
- Support CI/CD usage through GitLab and reusable GitHub Actions integration.

---

## Technologies Used

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- OpenAPI / JSON / YAML processing

### Frontend

- React
- Vite
- JavaScript / TypeScript
- HTML / CSS

### DevOps and Tooling

- Docker
- Docker Compose
- GitLab CI/CD
- GitHub Actions

### LLM Integration

- Local or cloud LLM providers
- Configurable model settings and prompts
- LLM-generated failure explanations and suggested follow-up tests
- Chat interface provide many possibilities

---

## Project Purpose

Many APIs are described using contracts such as OpenAPI specifications, but real API behaviour can still drift from what the specification says. Manually checking this is slow, repetitive, and difficult to scale.

ContractGuard helps solve this by automatically turning API specifications into executable test cases. It allows developers to inspect the generated tests, run them against real APIs, and understand the results more easily.

The main goal is to support a practical API testing workflow:

1. Upload or provide an API specification.
2. Parse the specification into an internal representation.
3. Generate contract-based test cases.
4. Review or edit the generated tests.
5. Run the tests against a target API.
6. View pass/fail/skipped results.
7. Use LLM explanations to understand failures.


## GitHub Actions Integration

ContractGuard includes a reusable GitHub Actions workflow so that other developers can run contract testing in their own repositories.

This integration allows a developer to configure:

- API specification path
- Runtime mapping file
- API base URL
- Authentication secrets
- Test execution settings

The workflow can then parse the specification, generate tests, execute them, and expose the results through CI logs and artifacts.

---

## Limitations

ContractGuard has been tested with large real world specifications such as GitHub REST API and PayPal Checkout API. However, performance cannot be fully guaranteed for extremely large specifications with deeply nested schemas, complex references, or multiple layers of recursion.

Some real API test failures may also be caused by external conditions rather than ContractGuard itself, such as:

- Missing or invalid authentication
- Limited token permissions
- Sandbox account state
- Missing real resource IDs
- Endpoints that require chained workflows
- External API rate limits or service behaviour

The frontend was tested manually, but no automated frontend test suite was implemented.

---

## Contributors

| Name | Student ID |
|---|---|
| Chunyang Wang | 21115214 |
| Justin Siak | 22449184 |

---

## Supervisor

Dr. Graham Healy

---

## Project Context

This project was developed as part of the final year Computer Science project at Dublin City University.