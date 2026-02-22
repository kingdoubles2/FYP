import json
import yaml
from typing import Dict, Any, List, Optional

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
    result = spec
    for part in parts:
        result = result[part]

    return result


def resolve_schema(schema: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve $ref in schema.
    """
    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], spec)
        return resolve_schema(resolved, spec)

    resolved_schema = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            resolved_schema[key] = resolve_schema(value, spec)
        elif isinstance(value, list):
            resolved_schema[key] = [
                resolve_schema(item, spec) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            resolved_schema[key] = value

    return resolved_schema


def extract_parameters(
    parameters: List[Dict[str, Any]],
    spec: Dict[str, Any]
) -> (List[ParamIR], List[ParamIR], List[ParamIR]):
    path_params = []
    query_params = []
    header_params = []

    for param in parameters:
        if "$ref" in param:
            param = resolve_ref(param["$ref"], spec)

        schema = resolve_schema(param.get("schema", {}), spec)

        param_ir = ParamIR(
            name=param["name"],
            location=param["in"],
            required=param.get("required", False),
            schema=schema
        )

        if param["in"] == "path":
            path_params.append(param_ir)
        elif param["in"] == "query":
            query_params.append(param_ir)
        elif param["in"] == "header":
            header_params.append(param_ir)

    return path_params, query_params, header_params


def extract_request_schema(operation: Dict[str, Any], spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_body = operation.get("requestBody")
    if not request_body:
        return None

    if "$ref" in request_body:
        request_body = resolve_ref(request_body["$ref"], spec)

    content = request_body.get("content", {})
    json_content = content.get("application/json")
    if not json_content:
        return None

    schema = json_content.get("schema")
    if not schema:
        return None

    return resolve_schema(schema, spec)


def extract_response_schemas(operation: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    responses = operation.get("responses", {})
    response_schemas = {}

    for status_code, response in responses.items():
        if "$ref" in response:
            response = resolve_ref(response["$ref"], spec)

        content = response.get("content", {})
        json_content = content.get("application/json")

        if not json_content:
            response_schemas[status_code] = None
            continue

        schema = json_content.get("schema")
        if not schema:
            response_schemas[status_code] = None
            continue

        response_schemas[status_code] = resolve_schema(schema, spec)

    return response_schemas


def parse_openapi(spec_text: str) -> ParsedSpecIR:
    spec = load_spec(spec_text)

    if "openapi" not in spec:
        raise ValueError("Not a valid OpenAPI 3 specification")

    if "paths" not in spec:
        raise ValueError("OpenAPI spec missing 'paths'")

    title = spec.get("info", {}).get("title", "Unknown")
    version = spec.get("info", {}).get("version", "Unknown")

    endpoints: List[EndpointIR] = []

    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue

            operation_id = operation.get("operationId")

            parameters = operation.get("parameters", [])
            path_level_params = path_item.get("parameters", [])
            all_params = path_level_params + parameters

            path_params, query_params, header_params = extract_parameters(
                all_params, spec
            )

            request_schema = extract_request_schema(operation, spec)
            response_schemas = extract_response_schemas(operation, spec)

            endpoint_ir = EndpointIR(
                method=method.upper(),
                path=path,
                operation_id=operation_id,
                path_params=path_params,
                query_params=query_params,
                header_params=header_params,
                request_schema=request_schema,
                response_schemas=response_schemas
            )

            endpoints.append(endpoint_ir)

    return ParsedSpecIR(
        title=title,
        version=version,
        endpoints=endpoints
    )