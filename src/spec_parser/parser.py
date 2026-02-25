import json
from typing import Dict, Any, List, Optional, Tuple

import yaml

from .models import ParsedSpecIR, EndpointIR, ParamIR


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def load_spec(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def resolve_ref(ref: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve local $ref like '#/components/schemas/User'
    """
    if not ref.startswith("#/"):
        raise ValueError(f"Only local refs supported for now: {ref}")

    parts = ref.lstrip("#/").split("/")
    result: Any = spec
    for part in parts:
        result = result[part]

    if not isinstance(result, dict):
        raise ValueError(f"Resolved ref is not an object: {ref}")

    return result


def resolve_schema(schema: Any, spec: Dict[str, Any]) -> Any:
    """
    Recursively resolve $ref in schema. Returns a dict/list/primitive.
    """
    if not isinstance(schema, (dict, list)):
        return schema

    if isinstance(schema, list):
        return [resolve_schema(item, spec) for item in schema]

    # dict
    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], spec)
        return resolve_schema(resolved, spec)

    resolved_schema: Dict[str, Any] = {}
    for key, value in schema.items():
        resolved_schema[key] = resolve_schema(value, spec)
    return resolved_schema


def extract_parameters(
    parameters: List[Dict[str, Any]],
    spec: Dict[str, Any],
) -> Tuple[List[ParamIR], List[ParamIR], List[ParamIR]]:
    path_params: List[ParamIR] = []
    query_params: List[ParamIR] = []
    header_params: List[ParamIR] = []

    for param in parameters:
        if not isinstance(param, dict):
            continue

        if "$ref" in param:
            param = resolve_ref(param["$ref"], spec)

        # Only handle standard param locations
        location = param.get("in")
        if location not in {"path", "query", "header"}:
            continue

        schema = resolve_schema(param.get("schema", {}), spec)

        param_ir = ParamIR(
            name=param["name"],
            location=location,
            required=param.get("required", False),
            schema=schema if isinstance(schema, dict) else {},
        )

        if location == "path":
            path_params.append(param_ir)
        elif location == "query":
            query_params.append(param_ir)
        elif location == "header":
            header_params.append(param_ir)

    return path_params, query_params, header_params


def extract_request_schema(operation: Dict[str, Any], spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_body = operation.get("requestBody")
    if not request_body:
        return None

    if isinstance(request_body, dict) and "$ref" in request_body:
        request_body = resolve_ref(request_body["$ref"], spec)

    if not isinstance(request_body, dict):
        return None

    content = request_body.get("content", {})
    json_content = content.get("application/json")
    if not isinstance(json_content, dict):
        return None

    schema = json_content.get("schema")
    if not schema:
        return None

    resolved = resolve_schema(schema, spec)
    return resolved if isinstance(resolved, dict) else None


def extract_response_schemas(operation: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    responses = operation.get("responses", {})
    response_schemas: Dict[str, Optional[Dict[str, Any]]] = {}

    if not isinstance(responses, dict):
        return response_schemas

    for status_code, response in responses.items():
        if isinstance(response, dict) and "$ref" in response:
            response = resolve_ref(response["$ref"], spec)

        if not isinstance(response, dict):
            response_schemas[str(status_code)] = None
            continue

        content = response.get("content", {})
        json_content = content.get("application/json")

        if not isinstance(json_content, dict):
            response_schemas[str(status_code)] = None
            continue

        schema = json_content.get("schema")
        if not schema:
            response_schemas[str(status_code)] = None
            continue

        resolved = resolve_schema(schema, spec)
        response_schemas[str(status_code)] = resolved if isinstance(resolved, dict) else None

    return response_schemas


def parse_openapi(spec_text: str) -> ParsedSpecIR:
    spec = load_spec(spec_text)

    if "openapi" not in spec:
        raise ValueError("Not a valid OpenAPI 3 specification (missing 'openapi')")

    if "paths" not in spec:
        raise ValueError("OpenAPI spec missing 'paths'")

    title = spec.get("info", {}).get("title", "Unknown")
    version = spec.get("info", {}).get("version", "Unknown")

    endpoints: List[EndpointIR] = []

    paths = spec["paths"]
    if not isinstance(paths, dict):
        raise ValueError("'paths' must be an object")

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # path-level parameters
        path_level_params = path_item.get("parameters", [])
        if not isinstance(path_level_params, list):
            path_level_params = []

        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                continue

            method_upper = method.upper()
            endpoint_id = f"{method_upper} {path}"
            operation_id = operation.get("operationId")

            op_params = operation.get("parameters", [])
            if not isinstance(op_params, list):
                op_params = []

            all_params = path_level_params + op_params

            path_params, query_params, header_params = extract_parameters(all_params, spec)
            request_schema = extract_request_schema(operation, spec)
            response_schemas = extract_response_schemas(operation, spec)

            endpoint_ir = EndpointIR(
                endpoint_id=endpoint_id,
                method=method_upper,
                path=path,
                operation_id=operation_id,
                path_params=path_params,
                query_params=query_params,
                header_params=header_params,
                request_schema=request_schema,
                response_schemas=response_schemas,
            )
            endpoints.append(endpoint_ir)

    return ParsedSpecIR(title=title, version=version, endpoints=endpoints)
