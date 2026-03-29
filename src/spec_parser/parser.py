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


def _first_example_value(examples: Dict[str, Any], spec: Dict[str, Any]) -> Any:
    """Return the ``value`` from the first entry of an OpenAPI examples map.

    Handles ``$ref`` inside individual example objects.  Returns a sentinel
    ``_MISSING`` when nothing usable is found.
    """
    for _key, example_obj in examples.items():
        if not isinstance(example_obj, dict):
            continue
        if "$ref" in example_obj:
            example_obj = resolve_ref(example_obj["$ref"], spec)
        if "value" in example_obj:
            return example_obj["value"]
    return _MISSING


_MISSING = object()  # sentinel – never appears in user data


def _pick_json_media(content: Any) -> Optional[Dict[str, Any]]:
    """Pick the best JSON-like media object from a content map."""
    if not isinstance(content, dict):
        return None

    exact = content.get("application/json")
    if isinstance(exact, dict):
        return exact

    for media_type, media_obj in content.items():
        if isinstance(media_type, str) and "json" in media_type.lower() and isinstance(media_obj, dict):
            return media_obj
    return None


def _extract_media_examples(media_obj: Any, spec: Dict[str, Any]) -> List[Any]:
    """Collect example payloads from a media object.

    Priority: explicit ``examples`` map first, then singular ``example``.
    """
    out: List[Any] = []
    if not isinstance(media_obj, dict):
        return out

    examples = media_obj.get("examples")
    if isinstance(examples, dict):
        for _name, ex in examples.items():
            if not isinstance(ex, dict):
                continue
            if "$ref" in ex:
                ex = resolve_ref(ex["$ref"], spec)
            if isinstance(ex, dict) and "value" in ex:
                out.append(ex["value"])

    if "example" in media_obj:
        out.append(media_obj["example"])

    return out


def _normalize_security_requirements(raw: Any) -> List[Dict[str, List[str]]]:
    """Normalize OpenAPI security requirements into a stable list format."""
    normalized: List[Dict[str, List[str]]] = []
    if not isinstance(raw, list):
        return normalized

    for req in raw:
        if not isinstance(req, dict):
            continue
        item: Dict[str, List[str]] = {}
        for scheme_name, scopes in req.items():
            if isinstance(scopes, list):
                item[str(scheme_name)] = [str(s) for s in scopes]
            else:
                item[str(scheme_name)] = []
        normalized.append(item)
    return normalized


def extract_operation_security(
    operation: Dict[str, Any],
    path_item: Dict[str, Any],
    spec: Dict[str, Any],
) -> List[Dict[str, List[str]]]:
    """Resolve effective security at operation/path/global level."""
    if "security" in operation:
        return _normalize_security_requirements(operation.get("security"))
    if "security" in path_item:
        return _normalize_security_requirements(path_item.get("security"))
    return _normalize_security_requirements(spec.get("security"))


def extract_security_schemes(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and resolve components.securitySchemes for auth-aware test generation."""
    components = spec.get("components")
    if not isinstance(components, dict):
        return {}

    schemes = components.get("securitySchemes")
    if not isinstance(schemes, dict):
        return {}

    out: Dict[str, Any] = {}
    for name, scheme in schemes.items():
        if isinstance(scheme, dict):
            out[str(name)] = resolve_schema(scheme, spec)
    return out


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
        param_schema = schema if isinstance(schema, dict) else {}

        # Embed parameter-level example into schema so the generator picks
        # it up via the existing example > default > enum priority chain.
        # Priority: parameter.examples[*].value > parameter.example
        if "example" not in param_schema:
            examples = param.get("examples")
            if isinstance(examples, dict) and examples:
                val = _first_example_value(examples, spec)
                if val is not _MISSING:
                    param_schema = dict(param_schema)
                    param_schema["example"] = val
            elif "example" in param:
                param_schema = dict(param_schema)
                param_schema["example"] = param["example"]

        param_ir = ParamIR(
            name=param["name"],
            location=location,
            required=param.get("required", False),
            schema=param_schema,
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
    json_content = _pick_json_media(content)
    if not isinstance(json_content, dict):
        return None

    schema = json_content.get("schema")
    if not schema:
        return None

    resolved = resolve_schema(schema, spec)
    if not isinstance(resolved, dict):
        return None

    # Embed request-body-level example into the schema so the generator
    # uses the realistic payload for happy path tests.
    # Priority: content.examples[*].value > content.example
    if "example" not in resolved:
        examples = json_content.get("examples")
        if isinstance(examples, dict) and examples:
            val = _first_example_value(examples, spec)
            if val is not _MISSING:
                resolved = dict(resolved)
                resolved["example"] = val
        elif "example" in json_content:
            resolved = dict(resolved)
            resolved["example"] = json_content["example"]

    return resolved


def extract_request_body_required(operation: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """Return whether requestBody is required for this operation."""
    request_body = operation.get("requestBody")
    if not request_body:
        return False

    if isinstance(request_body, dict) and "$ref" in request_body:
        request_body = resolve_ref(request_body["$ref"], spec)

    if not isinstance(request_body, dict):
        return False

    return bool(request_body.get("required", False))


def extract_request_examples(operation: Dict[str, Any], spec: Dict[str, Any]) -> List[Any]:
    """Extract requestBody examples from JSON media content."""
    request_body = operation.get("requestBody")
    if not request_body:
        return []

    if isinstance(request_body, dict) and "$ref" in request_body:
        request_body = resolve_ref(request_body["$ref"], spec)

    if not isinstance(request_body, dict):
        return []

    content = request_body.get("content", {})
    media = _pick_json_media(content)
    return _extract_media_examples(media, spec)


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
        json_content = _pick_json_media(content)

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


def extract_response_descriptions(operation: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, str]:
    """Extract per-status response description text."""
    responses = operation.get("responses", {})
    descriptions: Dict[str, str] = {}
    if not isinstance(responses, dict):
        return descriptions

    for status_code, response in responses.items():
        if isinstance(response, dict) and "$ref" in response:
            response = resolve_ref(response["$ref"], spec)
        if not isinstance(response, dict):
            continue
        desc = response.get("description")
        if isinstance(desc, str) and desc.strip():
            descriptions[str(status_code)] = desc

    return descriptions


def extract_response_examples(operation: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract per-status response examples from JSON media content."""
    responses = operation.get("responses", {})
    examples_by_status: Dict[str, List[Any]] = {}
    if not isinstance(responses, dict):
        return examples_by_status

    for status_code, response in responses.items():
        if isinstance(response, dict) and "$ref" in response:
            response = resolve_ref(response["$ref"], spec)
        if not isinstance(response, dict):
            continue

        media = _pick_json_media(response.get("content", {}))
        examples = _extract_media_examples(media, spec)
        if examples:
            examples_by_status[str(status_code)] = examples

    return examples_by_status


def _response_example_value(operation: Dict[str, Any], spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to extract a concrete example object from the 200/201 response.

    Returns the example value dict if found, else None.
    """
    responses = operation.get("responses", {})
    for code in ("200", "201"):
        resp = responses.get(code)
        if not isinstance(resp, dict):
            continue
        if "$ref" in resp:
            resp = resolve_ref(resp["$ref"], spec)
        content = resp.get("content", {})
        json_ct = _pick_json_media(content)
        if not isinstance(json_ct, dict):
            continue
        # Try examples map first, then singular example
        examples = json_ct.get("examples")
        if isinstance(examples, dict) and examples:
            val = _first_example_value(examples, spec)
            if val is not _MISSING and isinstance(val, dict):
                return val
        example = json_ct.get("example")
        if isinstance(example, dict):
            return example
    return None


def _flatten_response_example(resp_example: Dict[str, Any]) -> Dict[str, Any]:
    """Build a flat lookup of scalar fields from a response example.

    Checks the top level first, then looks inside the first element of any
    top-level array (common in paginated list responses like ``{"data": [...]}``)
    """
    flat: Dict[str, Any] = {}
    # Top-level scalar fields
    for k, v in resp_example.items():
        if not isinstance(v, (dict, list)):
            flat[k] = v
    # First element of any top-level array
    for _k, v in resp_example.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for ik, iv in v[0].items():
                if ik not in flat and not isinstance(iv, (dict, list)):
                    flat[ik] = iv
            break  # only use the first array field
    return flat


def _backfill_param_examples(
    path_params: List[ParamIR],
    query_params: List[ParamIR],
    operation: Dict[str, Any],
    spec: Dict[str, Any],
) -> None:
    """Fill missing param examples from the success response example.

    If the 200/201 response example contains a field whose name matches
    a parameter that has no example yet, use that value.
    Works for both path params (top-level match) and query params
    (top-level or inside first array element for list endpoints).
    """
    needs_example = [p for p in path_params + query_params if "example" not in p.schema]
    if not needs_example:
        return

    resp_example = _response_example_value(operation, spec)
    if not resp_example:
        return

    flat = _flatten_response_example(resp_example)

    for param_ir in needs_example:
        val = flat.get(param_ir.name)
        if val is not None:
            param_ir.schema = dict(param_ir.schema)
            param_ir.schema["example"] = val


def _extract_first_server_url(servers: Any) -> Optional[str]:
    """Return the URL from the first valid entry in an OpenAPI servers list."""
    if not isinstance(servers, list):
        return None
    for srv in servers:
        if isinstance(srv, dict) and srv.get("url"):
            return srv["url"]
    return None


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
            summary = operation.get("summary") if isinstance(operation.get("summary"), str) else None
            description = operation.get("description") if isinstance(operation.get("description"), str) else None

            op_params = operation.get("parameters", [])
            if not isinstance(op_params, list):
                op_params = []

            all_params = path_level_params + op_params

            path_params, query_params, header_params = extract_parameters(all_params, spec)
            _backfill_param_examples(path_params, query_params, operation, spec)
            request_schema = extract_request_schema(operation, spec)
            request_body_required = extract_request_body_required(operation, spec)
            request_examples = extract_request_examples(operation, spec)
            security = extract_operation_security(operation, path_item, spec)
            response_schemas = extract_response_schemas(operation, spec)
            response_descriptions = extract_response_descriptions(operation, spec)
            response_examples = extract_response_examples(operation, spec)

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
                summary=summary,
                description=description,
                request_body_required=request_body_required,
                request_examples=request_examples,
                security=security,
                response_descriptions=response_descriptions,
                response_examples=response_examples,
            )
            endpoints.append(endpoint_ir)

    # Extract base URL from OpenAPI servers field.
    # OpenAPI 3 allows servers at global, path, and operation levels.
    # We check global first, then fall back to the first path/operation server.
    base_url: Optional[str] = _extract_first_server_url(spec.get("servers"))
    if not base_url:
        for _path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            base_url = _extract_first_server_url(path_item.get("servers"))
            if base_url:
                break
            for _method, op in path_item.items():
                if isinstance(op, dict):
                    base_url = _extract_first_server_url(op.get("servers"))
                    if base_url:
                        break
            if base_url:
                break

    security_schemes = extract_security_schemes(spec)

    return ParsedSpecIR(
        title=title,
        version=version,
        base_url=base_url,
        endpoints=endpoints,
        security_schemes=security_schemes,
    )
