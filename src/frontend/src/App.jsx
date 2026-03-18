import { useEffect, useMemo, useState } from "react";
import {
  API_BASE_URL,
  clearSpecs,
  healthCheck,
  listSpecs,
  loginUser,
  registerUser,
  runGeneratedTests,
  uploadSpecFile,
} from "./api.js";
import { buildSpecPreview } from "./testPreview.js";

const SESSION_KEY = "contractguard.session.v1";
const SPEC_CACHE_KEY = "contractguard.spec-cache.v1";
const SPEC_FILE_EXTENSIONS = [".json", ".yaml", ".yml"];

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

function cloneJsonValue(value) {
  if (value === undefined) {
    return undefined;
  }

  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

function isSupportedSpecFile(file) {
  const name = String(file?.name || "").toLowerCase();
  return SPEC_FILE_EXTENSIONS.some((extension) => name.endsWith(extension));
}

function isLikelyInvalidSpecError(error) {
  const message = String(error?.message || "").toLowerCase();
  return (
    message.includes("openapi") ||
    message.includes("missing 'paths'") ||
    message.includes("missing paths") ||
    message.includes("invalid json") ||
    message.includes("invalid yaml") ||
    message.includes("yaml")
  );
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

function deepEqual(left, right) {
  if (left === right) {
    return true;
  }

  if (Number.isNaN(left) && Number.isNaN(right)) {
    return true;
  }

  if (typeof left !== typeof right) {
    return false;
  }

  if (left === null || right === null) {
    return left === right;
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      return false;
    }

    return left.every((value, index) => deepEqual(value, right[index]));
  }

  if (typeof left === "object") {
    const leftKeys = Object.keys(left);
    const rightKeys = Object.keys(right || {});

    if (leftKeys.length !== rightKeys.length) {
      return false;
    }

    return leftKeys.every((key) => Object.prototype.hasOwnProperty.call(right, key) && deepEqual(left[key], right[key]));
  }

  return false;
}

function applyGeneratedTestEdit(generatedTests, testIndex, field, value) {
  return generatedTests.map((testCase, index) => {
    if (index !== testIndex) {
      return testCase;
    }

    if (field === "expected_result") {
      return {
        ...testCase,
        expected_result: value,
      };
    }

    const steps = Array.isArray(testCase.steps) ? [...testCase.steps] : [];
    const firstStep = steps[0] && typeof steps[0] === "object"
      ? { ...steps[0] }
      : { step_number: 1, action: "Execute request", input_data: {} };

    firstStep.input_data = value;
    steps[0] = firstStep;

    return {
      ...testCase,
      steps,
    };
  });
}

function applyGeneratedTestReset(generatedTests, originalGeneratedTests, testIndex) {
  if (!Array.isArray(originalGeneratedTests) || !originalGeneratedTests[testIndex]) {
    return generatedTests;
  }

  return generatedTests.map((testCase, index) => {
    if (index !== testIndex) {
      return testCase;
    }

    return cloneJsonValue(originalGeneratedTests[testIndex]);
  });
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
        <p className="muted">Select JSON/YAML files or pick a folder. Multiple uploads are supported.</p>
      </div>

      <label className="upload-dropzone">
        <input type="file" accept=".json,.yaml,.yml" multiple onChange={onUpload} disabled={loading} />
        <span>{loading ? "Uploading..." : "Choose API files"}</span>
        <small>Supports multiple files</small>
      </label>

      <label className="folder-upload-inline">
        <input
          type="file"
          multiple
          webkitdirectory=""
          directory=""
          mozdirectory=""
          onChange={onUpload}
          disabled={loading}
        />
        <span>{loading ? "Uploading..." : "Or choose a folder"}</span>
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

function SpecDetails({ entry, runState, onUpdateTestCase, onResetTestCase, onRunTests }) {
  const generatedCases = Array.isArray(entry?.generatedTests) ? entry.generatedTests : [];
  const originalCases = Array.isArray(entry?.originalGeneratedTests) ? entry.originalGeneratedTests : generatedCases;
  const [jsonDrafts, setJsonDrafts] = useState({});

  useEffect(() => {
    if (!entry) {
      setJsonDrafts({});
      return;
    }

    const nextDrafts = {};
    generatedCases.forEach((testCase, index) => {
      const firstStep = testCase?.steps?.[0] || null;
      nextDrafts[index] = {
        inputDataText: toPrettyJson(firstStep?.input_data ?? null),
        expectedResultText: toPrettyJson(testCase?.expected_result ?? null),
        inputDataError: "",
        expectedResultError: "",
      };
    });

    setJsonDrafts(nextDrafts);
  }, [entry?.id]);

  function handleJsonEdit(testIndex, field, nextText) {
    const textKey = field === "input_data" ? "inputDataText" : "expectedResultText";
    const errorKey = field === "input_data" ? "inputDataError" : "expectedResultError";

    let nextError = "";
    let parsedValue = null;

    try {
      parsedValue = JSON.parse(nextText);
      if (!parsedValue || typeof parsedValue !== "object" || Array.isArray(parsedValue)) {
        nextError = "Must be a JSON object.";
      }
    } catch {
      nextError = "Invalid JSON.";
    }

    setJsonDrafts((current) => ({
      ...current,
      [testIndex]: {
        ...(current[testIndex] || {}),
        [textKey]: nextText,
        [errorKey]: nextError,
      },
    }));

    if (!nextError && entry) {
      onUpdateTestCase?.(entry.id, testIndex, field, parsedValue);
    }
  }

  function handleResetCase(testIndex) {
    if (!entry) {
      return;
    }

    const originalCase = originalCases[testIndex];
    const firstStep = originalCase?.steps?.[0] || null;

    setJsonDrafts((current) => ({
      ...current,
      [testIndex]: {
        ...(current[testIndex] || {}),
        inputDataText: toPrettyJson(firstStep?.input_data ?? null),
        expectedResultText: toPrettyJson(originalCase?.expected_result ?? null),
        inputDataError: "",
        expectedResultError: "",
      },
    }));

    onResetTestCase?.(entry.id, testIndex);
  }

  function handleSummaryToggleNoScroll(event) {
    event.preventDefault();
    const details = event.currentTarget.parentElement;
    if (!details) {
      return;
    }

    const lockedY = window.scrollY;
    details.open = !details.open;

    // Re-apply on consecutive frames to neutralize browser scroll anchoring.
    requestAnimationFrame(() => {
      window.scrollTo({ top: lockedY });
      requestAnimationFrame(() => {
        window.scrollTo({ top: lockedY });
      });
    });
  }

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

  const casesByCategory = generatedCases.reduce((acc, testCase, index) => {
    const key = testCase?.category || "other";
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push({ testCase, index });
    return acc;
  }, {});
  const hasJsonDraftErrors = Object.values(jsonDrafts).some(
    (draft) => Boolean(draft?.inputDataError || draft?.expectedResultError),
  );
  const latestRunSummary = runState?.result?.summary || null;

  function buildRunPayloadFromDrafts() {
    const nextDrafts = {};
    const nextTestCases = [];
    let hasValidationError = false;

    generatedCases.forEach((testCase, testIndex) => {
      const draft = jsonDrafts[testIndex] || {};
      const firstStep = testCase?.steps?.[0] || null;
      const inputDataText = draft.inputDataText ?? toPrettyJson(firstStep?.input_data ?? null);
      const expectedResultText = draft.expectedResultText ?? toPrettyJson(testCase?.expected_result ?? null);

      let parsedInputData = null;
      let parsedExpectedResult = null;
      let inputDataError = "";
      let expectedResultError = "";

      try {
        parsedInputData = JSON.parse(inputDataText);
        if (!parsedInputData || typeof parsedInputData !== "object" || Array.isArray(parsedInputData)) {
          inputDataError = "Must be a JSON object.";
        }
      } catch {
        inputDataError = "Invalid JSON.";
      }

      try {
        parsedExpectedResult = JSON.parse(expectedResultText);
        if (!parsedExpectedResult || typeof parsedExpectedResult !== "object" || Array.isArray(parsedExpectedResult)) {
          expectedResultError = "Must be a JSON object.";
        }
      } catch {
        expectedResultError = "Invalid JSON.";
      }

      nextDrafts[testIndex] = {
        ...draft,
        inputDataText,
        expectedResultText,
        inputDataError,
        expectedResultError,
      };

      if (inputDataError || expectedResultError) {
        hasValidationError = true;
        return;
      }

      const nextCase = cloneJsonValue(testCase) || {};
      const nextSteps = Array.isArray(nextCase.steps) ? [...nextCase.steps] : [];
      const nextFirstStep = nextSteps[0] && typeof nextSteps[0] === "object"
        ? { ...nextSteps[0] }
        : { step_number: 1, action: "Execute request", input_data: {} };
      nextFirstStep.input_data = parsedInputData;
      nextSteps[0] = nextFirstStep;
      nextCase.steps = nextSteps;
      nextCase.expected_result = parsedExpectedResult;
      nextTestCases.push(nextCase);
    });

    if (hasValidationError) {
      setJsonDrafts((current) => ({ ...current, ...nextDrafts }));
      return null;
    }

    return {
      api_title: entry?.title || entry?.filename || "Generated Test Suite",
      api_version: entry?.version || "Unknown",
      base_url: entry?.generatedSuite?.base_url || entry?.parsed?.base_url || null,
      test_cases: nextTestCases,
    };
  }

  function handleRunTests() {
    if (!entry || !onRunTests) {
      return;
    }

    const runPayload = buildRunPayloadFromDrafts();
    if (!runPayload) {
      return;
    }

    onRunTests(entry.id, runPayload);
  }

  return (
    <section className="panel details-panel">
      <div className="panel-header">
        <p className="eyebrow">Spec Detail</p>
        <h2>{entry.title || entry.filename}</h2>
        <p className="muted">Review metadata, test coverage overview, and edit generated JSON before running tests.</p>
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
            <details className="expandable">
              <summary onClick={handleSummaryToggleNoScroll}>Test Coverage</summary>
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
                    <div className="preview-actions">
                      <button
                        type="button"
                        className="primary-button run-tests-button"
                        onClick={handleRunTests}
                        disabled={generatedCases.length === 0 || hasJsonDraftErrors || runState?.loading}
                      >
                        {runState?.loading ? "Running..." : "Run Tests"}
                      </button>
                      {hasJsonDraftErrors ? <p className="json-error">Fix invalid JSON before running tests.</p> : null}
                      {runState?.error ? <p className="message error">{runState.error}</p> : null}
                      {latestRunSummary ? (
                        <div className="run-summary">
                          <strong>Latest Run</strong>
                          <div className="run-summary-row">
                            <span className="run-chip run-chip-total">Total: {latestRunSummary.total ?? 0}</span>
                            <span className="run-chip run-chip-pass">Passed: {latestRunSummary.passed ?? 0}</span>
                            <span className="run-chip run-chip-fail">Failed: {latestRunSummary.failed ?? 0}</span>
                            <span className="run-chip run-chip-skip">Skipped: {latestRunSummary.skipped ?? 0}</span>
                          </div>
                        </div>
                      ) : null}
                    </div>
                    <div className="coverage-categories">
                      {Object.entries(entry.preview.totals).map(([category, count]) => {
                        const categoryCases = casesByCategory[category] || [];
                        return (
                          <details key={category} className="category-detail">
                            <summary onClick={handleSummaryToggleNoScroll}>
                              <span>{prettifyCategory(category)}</span>
                              <strong>{count}</strong>
                            </summary>
                            <div className="category-body">
                              {categoryCases.length > 0 ? (
                                <div className="testcase-list">
                                  {categoryCases.map(({ testCase, index: testIndex }) => {
                                    const firstStep = testCase.steps?.[0] || null;
                                    const inputData = firstStep?.input_data ?? null;
                                    const expectedResult = testCase.expected_result ?? null;
                                    const originalCase = originalCases[testIndex] || {};
                                    const originalFirstStep = originalCase?.steps?.[0] || null;
                                    const originalInputData = originalFirstStep?.input_data ?? null;
                                    const originalExpectedResult = originalCase?.expected_result ?? null;
                                    const draft = jsonDrafts[testIndex] || {};
                                    const inputDataText = draft.inputDataText ?? toPrettyJson(inputData);
                                    const expectedResultText = draft.expectedResultText ?? toPrettyJson(expectedResult);
                                    const inputDataError = draft.inputDataError || "";
                                    const expectedResultError = draft.expectedResultError || "";
                                    const originalInputDataText = toPrettyJson(originalInputData);
                                    const originalExpectedResultText = toPrettyJson(originalExpectedResult);
                                    const hasParsedEdits =
                                      !deepEqual(inputData, originalInputData) || !deepEqual(expectedResult, originalExpectedResult);
                                    const hasDraftEdits =
                                      inputDataText !== originalInputDataText || expectedResultText !== originalExpectedResultText;
                                    const shouldShowReset = hasParsedEdits || hasDraftEdits;

                                    return (
                                      <details
                                        key={testCase.test_id || `${testCase.title}-${testIndex}`}
                                        className="testcase-detail"
                                      >
                                        <summary onClick={handleSummaryToggleNoScroll}>
                                          <span>{testCase.test_id || "Test case"}</span>
                                          <span>{testCase.method} {testCase.path}</span>
                                        </summary>
                                        <div className="testcase-body">
                                          <p className="testcase-title">{testCase.title}</p>
                                          <p className="testcase-line">
                                            <strong>Action:</strong> {firstStep?.action || "Step details unavailable"}
                                          </p>

                                          <div className="json-section">
                                            <div className="json-section-header">
                                              <span className="json-label">Input Data</span>
                                              {shouldShowReset ? (
                                                <button
                                                  type="button"
                                                  className="reset-text-button"
                                                  onClick={() => handleResetCase(testIndex)}
                                                >
                                                  Reset
                                                </button>
                                              ) : null}
                                            </div>
                                            <textarea
                                              className="json-editor"
                                              value={inputDataText}
                                              onChange={(event) => handleJsonEdit(testIndex, "input_data", event.target.value)}
                                              rows={8}
                                              spellCheck={false}
                                            />
                                            {inputDataError ? <p className="json-error">{inputDataError}</p> : null}
                                          </div>

                                          <div className="json-section">
                                            <span className="json-label">Expected Result</span>
                                            <textarea
                                              className="json-editor"
                                              value={expectedResultText}
                                              onChange={(event) => handleJsonEdit(testIndex, "expected_result", event.target.value)}
                                              rows={8}
                                              spellCheck={false}
                                            />
                                            {expectedResultError ? <p className="json-error">{expectedResultError}</p> : null}
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

            <details className="expandable">
              <summary onClick={handleSummaryToggleNoScroll}>Endpoint Breakdown ({entry.preview.endpoints.length})</summary>
              <div className="expandable-body">
                <div className="endpoint-list">
                  {entry.preview.endpoints.map((endpoint) => (
                    <details key={endpoint.endpointId} className="endpoint-card">
                      <summary onClick={handleSummaryToggleNoScroll}>
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
  const [testRunBySpecId, setTestRunBySpecId] = useState({});

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
      setTestRunBySpecId({});
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
              const generatedSuite = cached?.generatedSuite && typeof cached.generatedSuite === "object"
                ? cached.generatedSuite
                : null;
              const suiteTestCases = Array.isArray(generatedSuite?.test_cases) ? generatedSuite.test_cases : [];
              const generatedTests = Array.isArray(cached?.generatedTests) ? cached.generatedTests : suiteTestCases;
              const originalGeneratedTests = Array.isArray(cached?.originalGeneratedTests)
                ? cached.originalGeneratedTests
                : generatedTests;
              return {
                ...row,
                created_at: row.created_at || cached?.createdAt || cached?.uploadedAt || null,
                preview: cached?.preview || null,
                parsed: cached?.parsed || null,
                generatedSuite,
                generatedTests,
                originalGeneratedTests,
                totalCases: cached?.preview?.totalCases || generatedTests.length || 0,
              };
            })
          : [];

        setSpecHistory(merged);
        setSelectedSpecId((current) => current ?? merged[0]?.id ?? null);
        setTestRunBySpecId((current) => {
          const validIds = new Set(merged.map((entry) => String(entry.id)));
          const next = {};
          for (const [specId, value] of Object.entries(current)) {
            if (validIds.has(String(specId))) {
              next[specId] = value;
            }
          }
          return next;
        });
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
  }, [session]);

  const selectedEntry = useMemo(
    () => specHistory.find((entry) => entry.id === selectedSpecId) || null,
    [selectedSpecId, specHistory],
  );
  const selectedRunState = useMemo(() => {
    if (!selectedEntry) {
      return { loading: false, error: "", result: null };
    }
    return testRunBySpecId[selectedEntry.id] || { loading: false, error: "", result: null };
  }, [selectedEntry, testRunBySpecId]);

  function handleUpdateTestCase(specId, testIndex, field, value) {
    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }

        const tests = Array.isArray(entry.generatedTests) ? entry.generatedTests : [];
        const updatedTests = applyGeneratedTestEdit(tests, testIndex, field, value);
        return {
          ...entry,
          generatedTests: updatedTests,
          generatedSuite: entry.generatedSuite
            ? { ...entry.generatedSuite, test_cases: updatedTests }
            : entry.generatedSuite,
        };
      }),
    );
    setTestRunBySpecId((current) => {
      if (!current[specId]) {
        return current;
      }
      return {
        ...current,
        [specId]: { loading: false, error: "", result: null },
      };
    });

    if (!session?.userId) {
      return;
    }

    setSpecCache((current) => {
      const userCache = { ...(current[session.userId] || {}) };
      const existing = userCache[specId];
      if (!existing) {
        return current;
      }

      const currentTests = Array.isArray(existing.generatedTests) ? existing.generatedTests : [];
      const updatedTests = applyGeneratedTestEdit(currentTests, testIndex, field, value);
      userCache[specId] = {
        ...existing,
        generatedTests: updatedTests,
        generatedSuite: existing.generatedSuite
          ? { ...existing.generatedSuite, test_cases: updatedTests }
          : existing.generatedSuite,
        originalGeneratedTests: Array.isArray(existing.originalGeneratedTests)
          ? existing.originalGeneratedTests
          : cloneJsonValue(currentTests),
      };

      return {
        ...current,
        [session.userId]: userCache,
      };
    });
  }

  function handleResetTestCase(specId, testIndex) {
    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }

        const tests = Array.isArray(entry.generatedTests) ? entry.generatedTests : [];
        const originalTests = Array.isArray(entry.originalGeneratedTests) ? entry.originalGeneratedTests : tests;
        const resetTests = applyGeneratedTestReset(tests, originalTests, testIndex);
        return {
          ...entry,
          generatedTests: resetTests,
          generatedSuite: entry.generatedSuite
            ? { ...entry.generatedSuite, test_cases: resetTests }
            : entry.generatedSuite,
        };
      }),
    );
    setTestRunBySpecId((current) => {
      if (!current[specId]) {
        return current;
      }
      return {
        ...current,
        [specId]: { loading: false, error: "", result: null },
      };
    });

    if (!session?.userId) {
      return;
    }

    setSpecCache((current) => {
      const userCache = { ...(current[session.userId] || {}) };
      const existing = userCache[specId];
      if (!existing) {
        return current;
      }

      const currentTests = Array.isArray(existing.generatedTests) ? existing.generatedTests : [];
      const originalTests = Array.isArray(existing.originalGeneratedTests)
        ? existing.originalGeneratedTests
        : currentTests;
      const resetTests = applyGeneratedTestReset(currentTests, originalTests, testIndex);

      userCache[specId] = {
        ...existing,
        generatedTests: resetTests,
        generatedSuite: existing.generatedSuite
          ? { ...existing.generatedSuite, test_cases: resetTests }
          : existing.generatedSuite,
        originalGeneratedTests: originalTests,
      };

      return {
        ...current,
        [session.userId]: userCache,
      };
    });
  }

  async function handleRunTests(specId, suitePayload) {
    if (!session?.token) {
      return;
    }

    setTestRunBySpecId((current) => ({
      ...current,
      [specId]: {
        loading: true,
        error: "",
        result: current[specId]?.result || null,
      },
    }));

    try {
      const payload = await runGeneratedTests(session.token, suitePayload);
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          loading: false,
          error: "",
          result: payload || null,
        },
      }));
    } catch (error) {
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          loading: false,
          error: error.message,
          result: current[specId]?.result || null,
        },
      }));
    }
  }

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
    setTestRunBySpecId({});
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
      setTestRunBySpecId({});
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
    const selectedFiles = Array.from(event.target.files || []);
    event.target.value = "";

    if (selectedFiles.length === 0 || !session?.token) {
      return;
    }

    const files = selectedFiles.filter(isSupportedSpecFile);
    const skippedCount = selectedFiles.length - files.length;

    if (files.length === 0) {
      setUploadError("No JSON/YAML OpenAPI files found in your selection.");
      setUploadMessage("");
      return;
    }

    setUploadLoading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const nextUserCache = { ...(specCache[session.userId] || {}) };
      let lastUploadedId = null;
      const uploadedEntries = [];
      let skippedInvalidSpecCount = 0;

      for (const file of files) {
        try {
          const payload = await uploadSpecFile(session.token, file);
          const parsed = payload?.parsed || null;
          const preview = parsed ? buildSpecPreview(parsed) : null;
          const generatedSuite = payload?.generated_tests && typeof payload.generated_tests === "object"
            ? cloneJsonValue(payload.generated_tests)
            : null;
          const rawGeneratedTests = Array.isArray(generatedSuite?.test_cases) ? generatedSuite.test_cases : [];
          const generatedTests = cloneJsonValue(rawGeneratedTests) || [];
          const originalGeneratedTests = cloneJsonValue(rawGeneratedTests) || [];
          const createdAt = new Date().toISOString();
          lastUploadedId = payload?.id ?? lastUploadedId;

          nextUserCache[payload.id] = {
            filename: file.name,
            uploadedAt: createdAt,
            createdAt,
            parsed,
            preview,
            generatedSuite,
            generatedTests,
            originalGeneratedTests,
          };

          uploadedEntries.push({
            id: payload.id,
            filename: file.name,
            title: parsed?.title || file.name,
            version: parsed?.version || "Unknown",
            created_at: createdAt,
            parsed,
            preview,
            generatedSuite,
            generatedTests,
            originalGeneratedTests,
            totalCases: preview?.totalCases || generatedTests.length || 0,
          });
        } catch (error) {
          if (isLikelyInvalidSpecError(error)) {
            skippedInvalidSpecCount += 1;
            continue;
          }
          throw error;
        }
      }

      if (uploadedEntries.length === 0) {
        const skippedTotal = skippedCount + skippedInvalidSpecCount;
        setUploadError(
          skippedTotal > 0
            ? `No valid OpenAPI specs were uploaded. Skipped ${skippedTotal} file${skippedTotal === 1 ? "" : "s"}.`
            : "No files were uploaded.",
        );
        setUploadMessage("");
        return;
      }

      setSpecCache((current) => ({
        ...current,
        [session.userId]: nextUserCache,
      }));
      setSpecHistory((current) => {
        const byId = new Map(current.map((entry) => [entry.id, entry]));
        for (const entry of uploadedEntries) {
          byId.set(entry.id, entry);
        }

        return Array.from(byId.values()).sort((a, b) => Number(b.id) - Number(a.id));
      });
      if (lastUploadedId !== null) {
        setSelectedSpecId(lastUploadedId);
      }

      const skippedTotal = skippedCount + skippedInvalidSpecCount;
      const skippedSuffix = skippedTotal > 0
        ? ` Skipped ${skippedTotal} unsupported/invalid file${skippedTotal === 1 ? "" : "s"}.`
        : "";
      setUploadMessage(
        `Uploaded ${uploadedEntries.length} valid spec file${uploadedEntries.length === 1 ? "" : "s"} successfully.${skippedSuffix}`,
      );
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
          <p className="eyebrow">API Contract Testing</p>
          <h1>ContractGuard</h1>
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

          <SpecDetails
            entry={selectedEntry}
            runState={selectedRunState}
            onUpdateTestCase={handleUpdateTestCase}
            onResetTestCase={handleResetTestCase}
            onRunTests={handleRunTests}
          />
        </main>
      )}
    </div>
  );
}
