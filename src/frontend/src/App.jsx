import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL, clearSpecs, healthCheck, listSpecs, loginUser, registerUser, uploadSpecFile } from "./api.js";
import { buildSpecPreview } from "./testPreview.js";

const SESSION_KEY = "contractguard.session.v1";
const SPEC_CACHE_KEY = "contractguard.spec-cache.v1";

function decodeJwtPayload(token) {
  if (!token) {
    return null;
  }

  try {
    const [, payload] = token.split(".");
    if (!payload) {
      return null;
    }

    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padding = normalized.length % 4 === 0 ? "" : "=".repeat(4 - (normalized.length % 4));
    return JSON.parse(atob(normalized + padding));
  } catch {
    return null;
  }
}

function buildSession(token, email) {
  const payload = decodeJwtPayload(token);
  return {
    token,
    email,
    userId: payload?.sub ? String(payload.sub) : "anonymous",
  };
}

function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw);
    if (!parsed?.token || !parsed?.email) {
      return null;
    }

    return buildSession(parsed.token, parsed.email);
  } catch {
    return null;
  }
}

function saveSession(session) {
  localStorage.setItem(
    SESSION_KEY,
    JSON.stringify({
      token: session.token,
      email: session.email,
    }),
  );
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function loadSpecCache() {
  try {
    const raw = localStorage.getItem(SPEC_CACHE_KEY);
    if (!raw) {
      return {};
    }

    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveSpecCache(cache) {
  localStorage.setItem(SPEC_CACHE_KEY, JSON.stringify(cache));
}

function formatDate(value) {
  if (!value) {
    return "Unknown";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function keepScrollOnToggle() {
  const currentY = window.scrollY;
  requestAnimationFrame(() => {
    window.scrollTo({ top: currentY });
  });
}

function toPrettyJson(value) {
  if (value === undefined) {
    return "null";
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function prettifyCategory(categoryKey) {
  return String(categoryKey || "").replaceAll("_", " ");
}

function StatCard({ label, value, accent }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className={`stat-value ${accent}`}>{value}</strong>
    </div>
  );
}

function AuthPanel({ mode, email, password, loading, error, onModeChange, onEmailChange, onPasswordChange, onSubmit }) {
  return (
    <section className="panel auth-panel">
      <div className="panel-header">
        <p className="eyebrow">Account Access</p>
        <h2>{mode === "login" ? "Sign in to your workspace" : "Create a user account"}</h2>
        <p className="muted">Sign in or create an account to manage your API specifications.</p>
      </div>

      <div className="tab-row" role="tablist" aria-label="Authentication mode">
        <button
          type="button"
          className={`tab-button ${mode === "login" ? "active" : ""}`}
          onClick={() => onModeChange("login")}
        >
          Login
        </button>
        <button
          type="button"
          className={`tab-button ${mode === "register" ? "active" : ""}`}
          onClick={() => onModeChange("register")}
        >
          Register
        </button>
      </div>

      <form className="auth-form" onSubmit={onSubmit}>
        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => onEmailChange(event.target.value)}
            placeholder="name@example.com"
            autoComplete="email"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => onPasswordChange(event.target.value)}
            placeholder="Enter your password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
          />
        </label>

        <button type="submit" className="primary-button" disabled={loading}>
          {loading ? "Working..." : mode === "login" ? "Login" : "Register and Login"}
        </button>
      </form>

      {error ? <p className="message error">{error}</p> : null}
    </section>
  );
}

function UploadPanel({ loading, onUpload, message, error }) {
  return (
    <section className="panel upload-panel">
      <div className="panel-header">
        <p className="eyebrow">Uploads</p>
        <h2>Upload one or more API specs</h2>
        <p className="muted">Select JSON or YAML OpenAPI files. Multiple uploads are supported.</p>
      </div>

      <label className="upload-dropzone">
        <input type="file" accept=".json,.yaml,.yml" multiple onChange={onUpload} disabled={loading} />
        <span>{loading ? "Uploading..." : "Choose API files"}</span>
        <small>Supports multiple selection</small>
      </label>

      {message ? <p className="message success">{message}</p> : null}
      {error ? <p className="message error">{error}</p> : null}
    </section>
  );
}

function HistoryList({ entries, selectedSpecId, onSelect, onClear, clearing }) {
  return (
    <section className="panel history-panel">
      <div className="panel-header">
        <div className="panel-header-row">
          <div>
            <p className="eyebrow">History</p>
            <h2>Your upload history</h2>
          </div>
          <button
            type="button"
            className="danger-button"
            onClick={onClear}
            disabled={clearing || entries.length === 0}
          >
            {clearing ? "Clearing..." : "Clear History"}
          </button>
        </div>
        <p className="muted">Only your account uploads are shown here.</p>
      </div>

      {entries.length === 0 ? (
        <div className="empty-state">
          <p>No uploads yet.</p>
          <span>Upload a spec to create the first history entry.</span>
        </div>
      ) : (
        <div className="history-list">
          {entries.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={`history-item ${selectedSpecId === entry.id ? "active" : ""}`}
              onClick={() => onSelect(entry.id)}
            >
              <div className="history-topline">
                <strong>{entry.title || entry.filename}</strong>
                <span>#{entry.id}</span>
              </div>
              <div className="history-meta">
                <span>{entry.filename}</span>
                <span>{entry.version || "Unknown version"}</span>
              </div>
              <div className="history-meta">
                <span>{formatDate(entry.created_at)}</span>
                <span>{entry.totalCases ? `${entry.totalCases} generated tests` : "Metadata only"}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function SpecDetails({ entry }) {
  if (!entry) {
    return (
      <section className="panel details-panel">
        <div className="empty-state">
          <p>Select an upload</p>
          <span>Choose an item from your history to inspect its parsed API structure and generated test preview.</span>
        </div>
      </section>
    );
  }

  const generatedCases = Array.isArray(entry.generatedTests) ? entry.generatedTests : [];
  const casesByCategory = generatedCases.reduce((acc, testCase) => {
    const key = testCase?.category || "other";
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(testCase);
    return acc;
  }, {});

  return (
    <section className="panel details-panel">
      <div className="panel-header">
        <p className="eyebrow">Spec Detail</p>
        <h2>{entry.title || entry.filename}</h2>
        <p className="muted">Review metadata, test coverage overview, and endpoint details.</p>
      </div>

      <div className="stats-grid">
        <StatCard label="Spec ID" value={entry.id} accent="accent-amber" />
        <StatCard label="Version" value={entry.version || "Unknown"} accent="accent-blue" />
        <StatCard label="Uploaded" value={formatDate(entry.created_at)} accent="accent-green" />
        <StatCard label="Endpoints" value={entry.preview?.endpointCount ?? "Unknown"} accent="accent-red" />
      </div>

      {entry.preview ? (
        <>
          <div className="expandable-list">
            <details className="expandable" onToggle={keepScrollOnToggle}>
              <summary>Test Coverage</summary>
              <div className="expandable-body">
                <div className="preview-grid">
                  <div className="subpanel">
                    <h3>Generated Test Preview</h3>
                    <div className="metrics-list">
                      <div className="metric-row">
                        <span>Total estimated tests</span>
                        <strong>{entry.preview.totalCases}</strong>
                      </div>
                    </div>
                    <div className="coverage-categories">
                      {Object.entries(entry.preview.totals).map(([category, count]) => {
                        const categoryCases = casesByCategory[category] || [];
                        return (
                          <details key={category} className="category-detail" onToggle={keepScrollOnToggle}>
                            <summary>
                              <span>{prettifyCategory(category)}</span>
                              <strong>{count}</strong>
                            </summary>
                            <div className="category-body">
                              {categoryCases.length > 0 ? (
                                <div className="testcase-list">
                                  {categoryCases.map((testCase) => {
                                    const firstStep = testCase.steps?.[0] || null;
                                    const inputData = firstStep?.input_data ?? null;
                                    const expectedResult = testCase.expected_result ?? null;

                                    return (
                                      <details
                                        key={testCase.test_id || testCase.title}
                                        className="testcase-detail"
                                        onToggle={keepScrollOnToggle}
                                      >
                                        <summary>
                                          <span>{testCase.test_id || "Test case"}</span>
                                          <span>{testCase.method} {testCase.path}</span>
                                        </summary>
                                        <div className="testcase-body">
                                          <p className="testcase-title">{testCase.title}</p>
                                          <p className="testcase-line">
                                            <strong>Action:</strong> {firstStep?.action || "Step details unavailable"}
                                          </p>

                                          <div className="json-section">
                                            <span className="json-label">Input Data (JSON)</span>
                                            <pre className="json-block">{toPrettyJson(inputData)}</pre>
                                          </div>

                                          <div className="json-section">
                                            <span className="json-label">Expected Result (JSON)</span>
                                            <pre className="json-block">{toPrettyJson(expectedResult)}</pre>
                                          </div>
                                        </div>
                                      </details>
                                    );
                                  })}
                                </div>
                              ) : (
                                <p className="muted">Detailed generated cases are not available for this category yet.</p>
                              )}
                            </div>
                          </details>
                        );
                      })}
                    </div>
                  </div>

                  <div className="subpanel">
                    <h3>Methods</h3>
                    <div className="category-list">
                      {Object.entries(entry.preview.methodCounts).map(([method, count]) => (
                        <div key={method} className="category-row">
                          <span>{method}</span>
                          <strong>{count}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </details>

            <details className="expandable" onToggle={keepScrollOnToggle}>
              <summary>Endpoint Breakdown ({entry.preview.endpoints.length})</summary>
              <div className="expandable-body">
                <div className="endpoint-list">
                  {entry.preview.endpoints.map((endpoint) => (
                    <details key={endpoint.endpointId} className="endpoint-card" onToggle={keepScrollOnToggle}>
                      <summary>
                        <span className={`method-badge method-${endpoint.method.toLowerCase()}`}>{endpoint.method}</span>
                        <code>{endpoint.path}</code>
                        <span className="endpoint-total">Estimated total: {endpoint.totalCases}</span>
                      </summary>
                      <div className="endpoint-body">
                        <div className="endpoint-subtitle">
                          <span>{endpoint.operationId}</span>
                          <span>{endpoint.responseCodes.join(", ") || "No response codes"}</span>
                        </div>
                        <div className="endpoint-categories">
                          {Object.entries(endpoint.byCategory)
                            .filter(([, count]) => count > 0)
                            .map(([category, count]) => (
                              <span key={category} className="mini-pill">
                                {category.replaceAll("_", " ")}: {count}
                              </span>
                            ))}
                        </div>
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            </details>
          </div>
        </>
      ) : (
        <div className="empty-state">
          <p>Detailed preview unavailable for this entry.</p>
          <span>
            Upload this specification from the current browser session to see expanded endpoint-level details.
          </span>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [session, setSession] = useState(() => loadSession());
  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking backend...");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [clearHistoryLoading, setClearHistoryLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [specHistory, setSpecHistory] = useState([]);
  const [specCache, setSpecCache] = useState(() => loadSpecCache());
  const [selectedSpecId, setSelectedSpecId] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function checkApi() {
      try {
        const payload = await healthCheck();
        if (!cancelled) {
          setApiStatus(payload?.ok ? "Backend online" : "Backend reachable");
        }
      } catch (error) {
        if (!cancelled) {
          setApiStatus(error.message);
        }
      }
    }

    checkApi();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    saveSpecCache(specCache);
  }, [specCache]);

  useEffect(() => {
    if (!session?.token) {
      setSpecHistory([]);
      setSelectedSpecId(null);
      return;
    }

    let cancelled = false;

    async function refreshHistory() {
      setHistoryLoading(true);
      setHistoryError("");

      try {
        const rows = await listSpecs(session.token);
        if (cancelled) {
          return;
        }

        const userCache = specCache[session.userId] || {};
        const merged = Array.isArray(rows)
          ? rows.map((row) => {
              const cached = userCache[row.id];
              const generatedTests = Array.isArray(cached?.generatedTests) ? cached.generatedTests : [];
              return {
                ...row,
                preview: cached?.preview || null,
                parsed: cached?.parsed || null,
                generatedTests,
                totalCases: cached?.preview?.totalCases || generatedTests.length || 0,
              };
            })
          : [];

        setSpecHistory(merged);
        setSelectedSpecId((current) => current ?? merged[0]?.id ?? null);
      } catch (error) {
        if (!cancelled) {
          setHistoryError(error.message);
          setSpecHistory([]);
        }
      } finally {
        if (!cancelled) {
          setHistoryLoading(false);
        }
      }
    }

    refreshHistory();

    return () => {
      cancelled = true;
    };
  }, [session, specCache]);

  const selectedEntry = useMemo(
    () => specHistory.find((entry) => entry.id === selectedSpecId) || null,
    [selectedSpecId, specHistory],
  );

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setAuthLoading(true);
    setAuthError("");

    try {
      if (authMode === "register") {
        await registerUser(email, password);
      }

      const loginPayload = await loginUser(email, password);
      const nextSession = buildSession(loginPayload.access_token, email);
      saveSession(nextSession);
      setSession(nextSession);
      setPassword("");
    } catch (error) {
      setAuthError(error.message);
    } finally {
      setAuthLoading(false);
    }
  }

  function handleLogout() {
    clearSession();
    setSession(null);
    setSpecHistory([]);
    setSelectedSpecId(null);
    setClearHistoryLoading(false);
    setUploadError("");
    setUploadMessage("");
  }

  async function handleClearHistory() {
    if (!session?.token || clearHistoryLoading || specHistory.length === 0) {
      return;
    }

    const confirmClear = window.confirm("Clear your upload history permanently?");
    if (!confirmClear) {
      return;
    }

    setClearHistoryLoading(true);
    setHistoryError("");

    try {
      await clearSpecs(session.token);
      setSpecHistory([]);
      setSelectedSpecId(null);
      setSpecCache((current) => {
        const next = { ...current };
        delete next[session.userId];
        return next;
      });
    } catch (error) {
      setHistoryError(error.message);
    } finally {
      setClearHistoryLoading(false);
    }
  }

  async function handleUpload(event) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";

    if (files.length === 0 || !session?.token) {
      return;
    }

    setUploadLoading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const nextUserCache = { ...(specCache[session.userId] || {}) };
      let lastUploadedId = null;

      for (const file of files) {
        const payload = await uploadSpecFile(session.token, file);
        const parsed = payload?.parsed || null;
        const preview = parsed ? buildSpecPreview(parsed) : null;
        const generatedTests = Array.isArray(payload?.generated_tests?.test_cases) ? payload.generated_tests.test_cases : [];
        lastUploadedId = payload?.id ?? lastUploadedId;

        nextUserCache[payload.id] = {
          filename: file.name,
          uploadedAt: new Date().toISOString(),
          parsed,
          preview,
          generatedTests,
        };
      }

      setSpecCache((current) => ({
        ...current,
        [session.userId]: nextUserCache,
      }));
      if (lastUploadedId !== null) {
        setSelectedSpecId(lastUploadedId);
      }

      setUploadMessage(`Uploaded ${files.length} file${files.length === 1 ? "" : "s"} successfully.`);
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setUploadLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="background-orb orb-one" />
      <div className="background-orb orb-two" />

      <header className="hero">
        <div>
          <p className="eyebrow">ContractGuard</p>
          <h1>API Contract Testing Dashboard</h1>
          <p className="hero-copy">
            Service status: <span>{apiStatus}</span>
          </p>
        </div>
        {session ? (
          <div className="session-card">
            <span>Signed in as</span>
            <strong>{session.email}</strong>
            <small>User #{session.userId}</small>
            <button type="button" className="secondary-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        ) : null}
      </header>

      {!session ? (
        <AuthPanel
          mode={authMode}
          email={email}
          password={password}
          loading={authLoading}
          error={authError}
          onModeChange={setAuthMode}
          onEmailChange={setEmail}
          onPasswordChange={setPassword}
          onSubmit={handleAuthSubmit}
        />
      ) : (
        <main className="dashboard-grid">
          <section className="panel summary-panel">
            <div className="panel-header">
              <p className="eyebrow">Workspace</p>
              <h2>Dashboard</h2>
              <p className="muted">Manage your uploads and inspect each API specification.</p>
            </div>

            <div className="stats-grid">
              <StatCard label="Uploads" value={historyLoading ? "..." : specHistory.length} accent="accent-blue" />
              <StatCard
                label="Cached previews"
                value={Object.keys(specCache[session.userId] || {}).length}
                accent="accent-green"
              />
              <StatCard
                label="Known tests"
                value={specHistory.reduce((sum, entry) => sum + (entry.totalCases || 0), 0)}
                accent="accent-red"
              />
              <StatCard label="API URL" value={API_BASE_URL} accent="accent-amber" />
            </div>

            {historyError ? <p className="message error">{historyError}</p> : null}
          </section>

          <div className="left-column">
            <UploadPanel loading={uploadLoading} onUpload={handleUpload} message={uploadMessage} error={uploadError} />
            <HistoryList
              entries={specHistory}
              selectedSpecId={selectedSpecId}
              onSelect={setSelectedSpecId}
              onClear={handleClearHistory}
              clearing={clearHistoryLoading}
            />
          </div>

          <SpecDetails entry={selectedEntry} />
        </main>
      )}
    </div>
  );
}
