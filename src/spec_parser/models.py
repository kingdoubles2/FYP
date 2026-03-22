from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

COMPACT_SCHEMA_MAX_DEPTH = 3
COMPACT_SCHEMA_MAX_PROPERTIES = 20
COMPACT_SCHEMA_MAX_ENUM = 20
COMPACT_SCHEMA_MAX_VARIANTS = 3


def _compact_schema(schema: Any, *, depth: int = 0) -> Any:
    if not isinstance(schema, dict):
        return schema

    if depth >= COMPACT_SCHEMA_MAX_DEPTH:
        schema_type = schema.get("type")
        if not schema_type and "properties" in schema:
            schema_type = "object"
        return {
            "type": schema_type,
            "x-contractguard-truncated": True,
            "x-contractguard-reason": "preview_depth",
        }

    compact: Dict[str, Any] = {}
    truncated = False

    for key in (
        "type",
        "format",
        "nullable",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "pattern",
    ):
        if key in schema:
            compact[key] = schema[key]

    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list):
        if len(enum_vals) > COMPACT_SCHEMA_MAX_ENUM:
            truncated = True
            compact["enum"] = enum_vals[:COMPACT_SCHEMA_MAX_ENUM]
        else:
            compact["enum"] = enum_vals

    if "items" in schema:
        compact["items"] = _compact_schema(schema["items"], depth=depth + 1)

    for keyword in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            if len(variants) > COMPACT_SCHEMA_MAX_VARIANTS:
                truncated = True
                variants = variants[:COMPACT_SCHEMA_MAX_VARIANTS]
            compact[keyword] = [_compact_schema(item, depth=depth + 1) for item in variants]

    properties = schema.get("properties")
    if isinstance(properties, dict):
        prop_items = list(properties.items())
        if len(prop_items) > COMPACT_SCHEMA_MAX_PROPERTIES:
            truncated = True
            prop_items = prop_items[:COMPACT_SCHEMA_MAX_PROPERTIES]
        compact["properties"] = {
            name: _compact_schema(prop_schema, depth=depth + 1)
            for name, prop_schema in prop_items
        }

        required = schema.get("required")
        if isinstance(required, list):
            allowed = set(compact["properties"].keys())
            required_filtered = [name for name in required if name in allowed]
            if required_filtered:
                compact["required"] = required_filtered

    if truncated:
        compact["x-contractguard-truncated"] = True
        compact["x-contractguard-reason"] = "preview_limits"

    return compact


@dataclass
class ParamIR:
    name: str
    location: str  # path, query, header
    required: bool
    schema: Dict[str, Any]

    def to_dict(self, *, compact: bool = False) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "required": self.required,
            "schema": _compact_schema(self.schema) if compact else self.schema,
        }


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

    def to_dict(self, *, compact: bool = False) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "method": self.method,
            "path": self.path,
            "operation_id": self.operation_id,
            "path_params": [param.to_dict(compact=compact) for param in self.path_params],
            "query_params": [param.to_dict(compact=compact) for param in self.query_params],
            "header_params": [param.to_dict(compact=compact) for param in self.header_params],
            "request_schema": _compact_schema(self.request_schema) if compact else self.request_schema,
            "response_schemas": {
                key: _compact_schema(value) if compact else value
                for key, value in self.response_schemas.items()
            },
        }


@dataclass
class ParsedSpecIR:
    title: str
    version: str
    base_url: Optional[str]
    endpoints: List[EndpointIR]

    def to_dict(self, *, compact: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "title": self.title,
            "version": self.version,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        d["endpoints"] = [ep.to_dict(compact=compact) for ep in self.endpoints]
        return d
