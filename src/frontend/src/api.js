const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  return text ? { detail: text } : null;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const payload = await readResponse(response);

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    if (payload && typeof payload === "object" && Object.prototype.hasOwnProperty.call(payload, "detail")) {
      const rawDetail = payload.detail;
      if (typeof rawDetail === "string" && rawDetail.trim()) {
        detail = rawDetail;
      } else if (rawDetail && typeof rawDetail === "object") {
        const message = typeof rawDetail.message === "string" ? rawDetail.message.trim() : "";
        const llmError = typeof rawDetail.llm_error === "string" ? rawDetail.llm_error.trim() : "";
        const failureKind = typeof rawDetail.failure_kind === "string" ? rawDetail.failure_kind.trim() : "";
        const provider = typeof rawDetail.provider === "string" ? rawDetail.provider.trim() : "";
        const model = typeof rawDetail.model === "string" ? rawDetail.model.trim() : "";
        const guidance = typeof rawDetail.guidance === "string" ? rawDetail.guidance.trim() : "";
        const diagnostics = Array.isArray(rawDetail.attempt_diagnostics)
          ? rawDetail.attempt_diagnostics.map((item) => String(item || "").trim()).filter(Boolean)
          : [];
        const chunks = [message || "Request failed"];
        if (failureKind) {
          chunks.push(`failure=${failureKind}`);
        }
        if (provider || model) {
          const target = [provider, model].filter(Boolean).join("/");
          if (target) {
            chunks.push(`target=${target}`);
          }
        }
        if (llmError) {
          chunks.push(`llm_error=${llmError}`);
        }
        if (diagnostics.length > 0) {
          chunks.push(`diagnostics=${diagnostics.join(" | ")}`);
        }
        if (guidance) {
          chunks.push(`guidance=${guidance}`);
        }
        detail = chunks.join(" ");
      } else if (rawDetail !== null && rawDetail !== undefined) {
        detail = String(rawDetail);
      }
    }
    throw new Error(detail);
  }

  return payload;
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function healthCheck() {
  return request("/api/health");
}

export async function registerUser(email, password) {
  return request("/api/auth/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email, password }),
  });
}

export async function loginUser(email, password) {
  return request("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email, password }),
  });
}

export async function listSpecs(token) {
  return request("/api/specs", {
    headers: {
      ...authHeaders(token),
    },
  });
}

export async function clearSpecs(token) {
  return request("/api/specs", {
    method: "DELETE",
    headers: {
      ...authHeaders(token),
    },
  });
}

export async function uploadSpecFile(token, file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/api/specs/parse", {
    method: "POST",
    headers: {
      ...authHeaders(token),
    },
    body: formData,
  });
}

export async function runGeneratedTests(token, suitePayload) {
  return request("/api/tests/run", {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(suitePayload),
  });
}

export async function fetchLatestRunForSpec(token, specId) {
  return request(`/api/specs/${encodeURIComponent(specId)}/runs/latest`, {
    headers: {
      ...authHeaders(token),
    },
  });
}

export async function requestLlmFailureAnalysis(token, runId, testId) {
  return request(`/api/tests/${encodeURIComponent(runId)}/cases/${encodeURIComponent(testId)}/llm/analyze-failure`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
    },
  });
}

export async function requestSpecGroundedChat(token, payload) {
  return request("/api/chat/spec-assistant", {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload || {}),
  });
}

export async function fetchLlmSettings(token) {
  return request("/api/llm/settings", {
    headers: {
      ...authHeaders(token),
    },
  });
}

export async function updateLlmSettings(token, payload) {
  return request("/api/llm/settings", {
    method: "PATCH",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload || {}),
  });
}

export async function addLlmModel(token, payload) {
  return request("/api/llm/settings/models", {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload || {}),
  });
}

export async function fetchLlmProviderModels(token, provider, payload) {
  return request(`/api/llm/settings/providers/${encodeURIComponent(provider)}/models`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload || {}),
  });
}

export async function deleteLlmModel(token, modelId) {
  return request(`/api/llm/settings/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
    headers: {
      ...authHeaders(token),
    },
  });
}

export { API_BASE_URL };
