from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class ParamIR:
    name: str
    location: str  # path, query, header
    required: bool
    schema: Dict[str, Any]


@dataclass
class EndpointIR:
    method: str
    path: str
    operation_id: Optional[str]
    path_params: List[ParamIR]
    query_params: List[ParamIR]
    header_params: List[ParamIR]
    request_schema: Optional[Dict[str, Any]]
    response_schemas: Dict[str, Optional[Dict[str, Any]]]


@dataclass
class ParsedSpecIR:
    title: str
    version: str
    endpoints: List[EndpointIR]