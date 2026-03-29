from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any


@dataclass
class ParamIR:
    name: str
    location: str  # path, query, header
    required: bool
    schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EndpointIR:
    endpoint_id: str  # stable key: "METHOD /path"
    method: str
    path: str
    operation_id: Optional[str]
    path_params: List[ParamIR]
    query_params: List[ParamIR]
    header_params: List[ParamIR]
    request_schema: Optional[Dict[str, Any]]
    response_schemas: Dict[str, Optional[Dict[str, Any]]]
    summary: Optional[str] = None
    description: Optional[str] = None
    request_body_required: bool = False
    request_examples: List[Any] = field(default_factory=list)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    response_descriptions: Dict[str, str] = field(default_factory=dict)
    response_examples: Dict[str, List[Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # dataclasses.asdict will also convert nested dataclasses
        return asdict(self)


@dataclass
class ParsedSpecIR:
    title: str
    version: str
    base_url: Optional[str]
    endpoints: List[EndpointIR]
    security_schemes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "title": self.title,
            "version": self.version,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        if self.security_schemes:
            d["security_schemes"] = self.security_schemes
        d["endpoints"] = [ep.to_dict() for ep in self.endpoints]
        return d