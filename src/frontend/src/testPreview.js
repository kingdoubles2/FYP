const AUTH_HEADER_NAMES = new Set(["api_key", "authorization", "x-api-key", "token", "bearer"]);
const INVALID_FORMATS = new Set(["date-time", "date", "email", "uri", "uuid"]);
const INTEGER_CONSTRAINTS = new Set(["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]);
const STRING_CONSTRAINTS = new Set(["minLength", "maxLength", "pattern", "format"]);
const ARRAY_CONSTRAINTS = new Set(["minItems", "maxItems"]);
const NUMBER_CONSTRAINTS = new Set(["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]);

function hasOwn(schema, key) {
  return Object.prototype.hasOwnProperty.call(schema || {}, key);
}

function getObjectKeys(value) {
  return value && typeof value === "object" ? Object.keys(value) : [];
}

function getSchemaType(schema) {
  if (!schema || typeof schema !== "object") {
    return "";
  }

  if (schema.type) {
    return schema.type;
  }

  if (schema.properties) {
    return "object";
  }

  return "";
}

function endpointHasExamples(endpoint) {
  for (const group of ["path_params", "query_params", "header_params"]) {
    for (const param of endpoint[group] || []) {
      if (hasOwn(param.schema || {}, "example")) {
        return true;
      }
    }
  }

  const requestSchema = endpoint.request_schema;
  if (!requestSchema || typeof requestSchema !== "object") {
    return false;
  }

  if (hasOwn(requestSchema, "example")) {
    return true;
  }

  for (const propertySchema of Object.values(requestSchema.properties || {})) {
    if (propertySchema && typeof propertySchema === "object" && hasOwn(propertySchema, "example")) {
      return true;
    }
  }

  return false;
}

function countHappyPath(endpoint) {
  let count = 1;

  if (endpointHasExamples(endpoint)) {
    count += 1;
  }

  for (const param of endpoint.query_params || []) {
    const enumValues = Array.isArray(param.schema?.enum) ? param.schema.enum : [];
    if (param.required && enumValues.length > 1) {
      count += Math.min(2, Math.max(0, enumValues.length - 1));
    }
  }

  return count;
}

function countNegativeDetails(endpoint) {
  let negativeType = 0;
  let negativeMissing = 0;
  let negativeInvalid = 0;

  negativeType += (endpoint.path_params || []).length;
  negativeType += (endpoint.query_params || []).filter((param) => param.required).length;

  negativeMissing += (endpoint.query_params || []).filter((param) => param.required).length;

  const requestSchema = endpoint.request_schema;
  const requestType = getSchemaType(requestSchema);
  if (requestSchema && requestType === "object") {
    negativeMissing += Array.isArray(requestSchema.required) ? requestSchema.required.length : 0;

    for (const fieldSchema of Object.values(requestSchema.properties || {})) {
      if (Array.isArray(fieldSchema?.enum) && fieldSchema.enum.length > 0) {
        negativeInvalid += 1;
      }
    }
  }

  for (const param of endpoint.query_params || []) {
    const paramSchema = param.schema || {};
    const enumValues = Array.isArray(paramSchema.enum)
      ? paramSchema.enum
      : Array.isArray(paramSchema.items?.enum)
        ? paramSchema.items.enum
        : [];

    if (enumValues.length > 0) {
      negativeInvalid += 1;
    }

    if (INVALID_FORMATS.has(paramSchema.format)) {
      negativeInvalid += 1;
    }
  }

  let total = negativeType + negativeMissing + negativeInvalid;
  if (total < 2) {
    negativeInvalid += 1;
    total += 1;
  }
  if (total < 2) {
    if (requestSchema) {
      negativeMissing += 1;
    } else {
      negativeInvalid += 1;
    }
  }

  return { negativeType, negativeMissing, negativeInvalid };
}

function hasExplicitConstraint(schema) {
  const type = getSchemaType(schema);
  const keys = new Set(getObjectKeys(schema));

  if (type === "integer") {
    return [...INTEGER_CONSTRAINTS].some((key) => keys.has(key)) || ["int32", "int64"].includes(schema.format);
  }

  if (type === "number") {
    return [...NUMBER_CONSTRAINTS].some((key) => keys.has(key));
  }

  if (type === "string") {
    return [...STRING_CONSTRAINTS].some((key) => keys.has(key));
  }

  if (type === "array") {
    return [...ARRAY_CONSTRAINTS].some((key) => keys.has(key));
  }

  return false;
}

function boundaryPairCount(schema) {
  const type = getSchemaType(schema);

  if (type === "integer") {
    let count = 0;
    if (hasOwn(schema, "minimum")) {
      count += 2;
    }
    if (hasOwn(schema, "maximum")) {
      count += 2;
    }
    if (!hasOwn(schema, "minimum") && !hasOwn(schema, "maximum") && ["int32", "int64"].includes(schema.format)) {
      count += 1;
    }
    return count;
  }

  if (type === "number") {
    let count = 0;
    if (hasOwn(schema, "minimum")) {
      count += 1;
    }
    if (hasOwn(schema, "maximum")) {
      count += 1;
    }
    return count;
  }

  if (type === "string") {
    let count = 0;
    if (typeof schema.minLength === "number" && schema.minLength > 0) {
      count += 2;
    }
    if (typeof schema.maxLength === "number") {
      count += 2;
    }
    return count;
  }

  if (type === "array") {
    let count = 0;
    if (typeof schema.minItems === "number" && schema.minItems > 0) {
      count += 1;
    }
    if (typeof schema.maxItems === "number") {
      count += 1;
    }
    return count;
  }

  return 0;
}

function countBoundary(endpoint) {
  const targets = [];

  for (const param of endpoint.path_params || []) {
    if (hasExplicitConstraint(param.schema || {})) {
      targets.push(param.schema);
    }
  }

  for (const param of endpoint.query_params || []) {
    if (hasExplicitConstraint(param.schema || {})) {
      targets.push(param.schema);
    }
  }

  const requestSchema = endpoint.request_schema;
  const requestType = getSchemaType(requestSchema);
  if (requestSchema && requestType === "object") {
    for (const fieldSchema of Object.values(requestSchema.properties || {})) {
      if (hasExplicitConstraint(fieldSchema || {})) {
        targets.push(fieldSchema);
      }
    }
  }

  let total = 0;
  for (const schema of targets) {
    total += boundaryPairCount(schema);
    if (total >= 5) {
      total = 5;
      break;
    }
  }

  if (total < 5 && requestSchema && requestType === "object") {
    const hasNullable = Object.values(requestSchema.properties || {}).some((fieldSchema) => fieldSchema?.nullable);
    if (hasNullable) {
      total += 1;
    }
  }

  return Math.min(total, 5);
}

function countAuth(endpoint) {
  let count = 0;
  for (const param of endpoint.header_params || []) {
    if (AUTH_HEADER_NAMES.has(String(param.name || "").toLowerCase())) {
      count += 3;
    }
  }
  return count;
}

function countErrorStatus(endpoint) {
  if (!endpoint.response_schemas || !hasOwn(endpoint.response_schemas, "404")) {
    return 0;
  }
  return (endpoint.path_params || []).length;
}

function summariseResponseCodes(endpoint) {
  return Object.keys(endpoint.response_schemas || {}).sort();
}

export function buildSpecPreview(parsedSpec) {
  const endpoints = Array.isArray(parsedSpec?.endpoints) ? parsedSpec.endpoints : [];

  const previewEndpoints = endpoints.map((endpoint) => {
    const happyPath = countHappyPath(endpoint);
    const negatives = countNegativeDetails(endpoint);
    const boundary = countBoundary(endpoint);
    const auth = countAuth(endpoint);
    const errorStatus = countErrorStatus(endpoint);

    const byCategory = {
      happy_path: happyPath,
      negative_type: negatives.negativeType,
      negative_missing: negatives.negativeMissing,
      negative_invalid: negatives.negativeInvalid,
      boundary,
      auth,
      error_status: errorStatus,
    };

    const totalCases = Object.values(byCategory).reduce((sum, value) => sum + value, 0);

    return {
      endpointId: endpoint.endpoint_id,
      method: endpoint.method,
      path: endpoint.path,
      operationId: endpoint.operation_id || "No operationId",
      responseCodes: summariseResponseCodes(endpoint),
      byCategory,
      totalCases,
    };
  });

  const totals = {
    happy_path: 0,
    negative_type: 0,
    negative_missing: 0,
    negative_invalid: 0,
    boundary: 0,
    auth: 0,
    error_status: 0,
  };

  const methodCounts = {};

  for (const endpoint of previewEndpoints) {
    methodCounts[endpoint.method] = (methodCounts[endpoint.method] || 0) + 1;
    for (const [category, count] of Object.entries(endpoint.byCategory)) {
      totals[category] += count;
    }
  }

  const totalCases = Object.values(totals).reduce((sum, value) => sum + value, 0);

  return {
    title: parsedSpec?.title || "Unknown API",
    version: parsedSpec?.version || "Unknown",
    endpointCount: previewEndpoints.length,
    totalCases,
    methodCounts,
    totals,
    endpoints: previewEndpoints,
  };
}
