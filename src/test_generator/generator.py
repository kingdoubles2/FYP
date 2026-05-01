from __future__ import annotations

import datetime
from typing import Any, Dict, List

from test_generator.models import TestSuite, TestCase
from test_generator.rules.happy_path import generate_happy_path_cases
from test_generator.rules.negative import generate_negative_cases
from test_generator.rules.boundary import generate_boundary_cases
from test_generator.rules.auth import generate_auth_cases
from test_generator.rules.error_status import generate_error_status_cases


def _make_path_slug(endpoint_id: str) -> str:
    """Convert 'GET /pet/{petId}' -> 'GET-pet-petId'."""
    method, path = endpoint_id.split(" ", 1)
    slug = path.strip("/").replace("/", "-").replace("{", "").replace("}", "")
    return f"{method}-{slug}"


def _deduplicate(cases: List[TestCase]) -> List[TestCase]:
    """Remove cases with duplicate titles, keeping the first occurrence."""
    seen: set[str] = set()
    unique: List[TestCase] = []
    for case in cases:
        if case.title not in seen:
            seen.add(case.title)
            unique.append(case)
    return unique


def generate_test_cases_for_endpoint(endpoint: Dict[str, Any]) -> List[TestCase]:
    """Run all rule generators for one endpoint, assign sequential test_ids."""
    slug = _make_path_slug(endpoint["endpoint_id"])

    cases: List[TestCase] = []
    cases.extend(generate_happy_path_cases(endpoint))
    cases.extend(generate_negative_cases(endpoint))
    cases.extend(generate_boundary_cases(endpoint))
    cases.extend(generate_auth_cases(endpoint))
    cases.extend(generate_error_status_cases(endpoint))

    cases = _deduplicate(cases)

    for i, case in enumerate(cases, start=1):
        case.test_id = f"TC-{slug}-{i:03d}"

    return cases


def generate_test_cases(ir_data: Dict[str, Any]) -> TestSuite:
    """Main entry point: full IR dict -> TestSuite."""
    all_cases: List[TestCase] = []

    for endpoint in ir_data.get("endpoints", []):
        all_cases.extend(generate_test_cases_for_endpoint(endpoint))

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return TestSuite(
        api_title=ir_data.get("title", ""),
        api_version=ir_data.get("version", ""),
        generated_at=timestamp,
        test_cases=all_cases,
        base_url=ir_data.get("base_url"),
    )
