from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any


@dataclass
class InputData:
    path_params: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestStep:
    step_number: int
    action: str
    input_data: InputData

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "input_data": self.input_data.to_dict(),
        }


@dataclass
class ExpectedResult:
    status_code: Optional[int] = None
    status_code_any_of: Optional[List[int]] = None
    description: str = ""
    response_body_contains: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.status_code is not None:
            d["status_code"] = self.status_code
        if self.status_code_any_of is not None:
            d["status_code_any_of"] = self.status_code_any_of
        d["description"] = self.description
        if self.response_body_contains is not None:
            d["response_body_contains"] = self.response_body_contains
        return d


@dataclass
class TestCase:
    test_id: str
    title: str
    category: str
    requirement_ref: str
    method: str
    path: str
    priority: str
    preconditions: List[str]
    steps: List[TestStep]
    expected_result: ExpectedResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "title": self.title,
            "category": self.category,
            "requirement_ref": self.requirement_ref,
            "method": self.method,
            "path": self.path,
            "priority": self.priority,
            "preconditions": self.preconditions,
            "steps": [s.to_dict() for s in self.steps],
            "expected_result": self.expected_result.to_dict(),
        }


@dataclass
class TestSuite:
    api_title: str
    api_version: str
    generated_at: str
    test_cases: List[TestCase]
    base_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "api_title": self.api_title,
            "api_version": self.api_version,
            "generated_at": self.generated_at,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        d["test_cases"] = [tc.to_dict() for tc in self.test_cases]
        return d
