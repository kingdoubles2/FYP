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
    const detail =
      payload && typeof payload === "object" && payload.detail
        ? payload.detail
        : `Request failed with status ${response.status}`;
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

export { API_BASE_URL };
