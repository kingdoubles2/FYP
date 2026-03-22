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
const THEME_KEY = "contractguard.theme.v1";
const SPEC_FILE_EXTENSIONS = [".json", ".yaml", ".yml"];
const JSON_EDITOR_INDENT = "  ";
const FAILURE_REASON_KEYS = ["message", "detail", "error", "reason", "title", "description"];

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

function getPreferredTheme() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "light";
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function loadTheme() {
  try {
    const raw = localStorage.getItem(THEME_KEY);
    if (raw === "light" || raw === "dark") {
      return raw;
    }
  } catch {
    return getPreferredTheme();
  }

  return getPreferredTheme();
}

function saveTheme(theme) {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Ignore write errors (private mode, storage restrictions).
  }
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

function normalizeRunOutcome(value) {
  const text = String(value || "").trim().toUpperCase();
  if (text === "PASS" || text === "FAIL" || text === "SKIP") {
    return text;
  }
  return "";
}

function formatExpectedStatusLabel(expectedStatus, expectedStatusAnyOf) {
  if (Array.isArray(expectedStatusAnyOf) && expectedStatusAnyOf.length > 0) {
    return expectedStatusAnyOf.join(" / ");
  }
  if (expectedStatus !== undefined && expectedStatus !== null && expectedStatus !== "") {
    return String(expectedStatus);
  }
  return "Unknown";
}

function isExpectedStatusMatch(actualStatus, expectedStatus, expectedStatusAnyOf) {
  if (actualStatus === undefined || actualStatus === null || actualStatus === "") {
    return false;
  }

  const normalizedActual = Number(actualStatus);
  if (!Number.isNaN(normalizedActual)) {
    if (Array.isArray(expectedStatusAnyOf) && expectedStatusAnyOf.length > 0) {
      return expectedStatusAnyOf.some((value) => Number(value) === normalizedActual);
    }
    if (expectedStatus !== undefined && expectedStatus !== null && expectedStatus !== "") {
      return Number(expectedStatus) === normalizedActual;
    }
  }

  const actualText = String(actualStatus);
  if (Array.isArray(expectedStatusAnyOf) && expectedStatusAnyOf.length > 0) {
    return expectedStatusAnyOf.some((value) => String(value) === actualText);
  }
  if (expectedStatus !== undefined && expectedStatus !== null && expectedStatus !== "") {
    return String(expectedStatus) === actualText;
  }
  return false;
}

function truncateText(text, maxLength = 220) {
  if (!text || text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

function parseResponseSnippetMeta(snippetText) {
  const raw = String(snippetText || "").trim();
  if (!raw) {
    return { reason: "", raw: "", documentationUrl: "" };
  }

  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const documentationUrl = String(parsed.documentation_url || parsed.docs_url || parsed.help_url || "").trim();

      for (const key of FAILURE_REASON_KEYS) {
        const value = parsed[key];
        if (typeof value === "string" && value.trim()) {
          return {
            reason: value.trim(),
            raw: JSON.stringify(parsed, null, 2),
            documentationUrl,
          };
        }
      }

      if (Array.isArray(parsed.errors) && parsed.errors.length > 0) {
        const first = parsed.errors[0];
        if (typeof first === "string" && first.trim()) {
          return {
            reason: first.trim(),
            raw: JSON.stringify(parsed, null, 2),
            documentationUrl,
          };
        }
        if (first && typeof first === "object") {
          for (const key of FAILURE_REASON_KEYS) {
            const value = first[key];
            if (typeof value === "string" && value.trim()) {
              return {
                reason: value.trim(),
                raw: JSON.stringify(parsed, null, 2),
                documentationUrl,
              };
            }
          }
        }
      }

      return { reason: "", raw: JSON.stringify(parsed, null, 2), documentationUrl };
    }

    if (typeof parsed === "string" && parsed.trim()) {
      return { reason: parsed.trim(), raw, documentationUrl: "" };
    }
  } catch {
    // Non-JSON snippet; fall through.
  }

  return { reason: "", raw, documentationUrl: "" };
}

function isAbsoluteHttpUrl(value) {
  const text = String(value || "").trim();
  return /^https?:\/\/\S+$/i.test(text);
}

function getLineStartIndex(text, index) {
  const clampedIndex = Math.max(0, index);
  const breakIndex = text.lastIndexOf("\n", Math.max(0, clampedIndex - 1));
  return breakIndex === -1 ? 0 : breakIndex + 1;
}

function getLineEndIndex(text, index) {
  const clampedIndex = Math.max(0, index);
  const breakIndex = text.indexOf("\n", clampedIndex);
  return breakIndex === -1 ? text.length : breakIndex;
}

function countOutdentCharacters(lineText) {
  if (lineText.startsWith("\t")) {
    return 1;
  }

  let removeCount = 0;
  while (removeCount < JSON_EDITOR_INDENT.length && lineText.charAt(removeCount) === " ") {
    removeCount += 1;
  }
  return removeCount;
}

function getDefaultBaseUrl(entry) {
  const generatedBaseUrl = String(entry?.generatedSuite?.base_url || "").trim();
  if (isAbsoluteHttpUrl(generatedBaseUrl)) {
    return generatedBaseUrl;
  }

  const parsedBaseUrl = String(entry?.parsed?.base_url || "").trim();
  if (isAbsoluteHttpUrl(parsedBaseUrl)) {
    return parsedBaseUrl;
  }

  return "";
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
          {entries.map((entry, index) => (
            <button
              key={entry.id}
              type="button"
              className={`history-item ${selectedSpecId === entry.id ? "active" : ""}`}
              onClick={() => onSelect(entry.id)}
            >
              <div className="history-topline">
                <strong>{entry.title || entry.filename}</strong>
                <span>#{index + 1}</span>
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
  const runBaselineCases = Array.isArray(runState?.baselineTests) ? runState.baselineTests : [];
  const baselineCases = runBaselineCases.length > 0 ? runBaselineCases : originalCases;
  const [jsonDrafts, setJsonDrafts] = useState({});
  const [baseUrlInput, setBaseUrlInput] = useState("");
  const [runAuthMode, setRunAuthMode] = useState("none");
  const [runBearerToken, setRunBearerToken] = useState("");
  const [runApiKeyValue, setRunApiKeyValue] = useState("");
  const [runApiKeyHeader, setRunApiKeyHeader] = useState("X-API-Key");

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

  useEffect(() => {
    if (!entry) {
      setBaseUrlInput("");
      setRunAuthMode("none");
      setRunBearerToken("");
      setRunApiKeyValue("");
      setRunApiKeyHeader("X-API-Key");
      return;
    }

    setBaseUrlInput(getDefaultBaseUrl(entry));
    setRunAuthMode("none");
    setRunBearerToken("");
    setRunApiKeyValue("");
    setRunApiKeyHeader("X-API-Key");
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

  function commitJsonEditorUpdate(event, testIndex, field, nextText, nextSelectionStart, nextSelectionEnd = nextSelectionStart) {
    const textarea = event.currentTarget;
    handleJsonEdit(testIndex, field, nextText);

    requestAnimationFrame(() => {
      textarea.focus();
      textarea.setSelectionRange(nextSelectionStart, nextSelectionEnd);
    });
  }

  function handleJsonEditorKeyDown(testIndex, field, event) {
    const { key, shiftKey, altKey, ctrlKey, metaKey } = event;

    if (key !== "Tab" && key !== "Enter") {
      return;
    }

    const textarea = event.currentTarget;
    const value = textarea.value;
    const selectionStart = textarea.selectionStart;
    const selectionEnd = textarea.selectionEnd;

    if (key === "Tab") {
      event.preventDefault();

      if (!shiftKey) {
        if (selectionStart === selectionEnd) {
          const nextValue = `${value.slice(0, selectionStart)}${JSON_EDITOR_INDENT}${value.slice(selectionEnd)}`;
          const nextCursor = selectionStart + JSON_EDITOR_INDENT.length;
          commitJsonEditorUpdate(event, testIndex, field, nextValue, nextCursor);
          return;
        }

        const blockStart = getLineStartIndex(value, selectionStart);
        const blockEnd = getLineEndIndex(value, selectionEnd);
        const selectedBlock = value.slice(blockStart, blockEnd);
        const lines = selectedBlock.split("\n");
        const indentedBlock = lines.map((line) => `${JSON_EDITOR_INDENT}${line}`).join("\n");
        const nextValue = `${value.slice(0, blockStart)}${indentedBlock}${value.slice(blockEnd)}`;
        const nextSelectionStart = selectionStart + JSON_EDITOR_INDENT.length;
        const nextSelectionEnd = selectionEnd + (JSON_EDITOR_INDENT.length * lines.length);
        commitJsonEditorUpdate(event, testIndex, field, nextValue, nextSelectionStart, nextSelectionEnd);
        return;
      }

      const blockStart = getLineStartIndex(value, selectionStart);
      const blockEnd = getLineEndIndex(value, selectionEnd);
      const selectedBlock = value.slice(blockStart, blockEnd);
      const lines = selectedBlock.split("\n");

      let lineGlobalStart = blockStart;
      let removedBeforeStart = 0;
      let removedBeforeEnd = 0;

      const outdentedLines = lines.map((line) => {
        const removeCount = countOutdentCharacters(line);
        const charsBeforeStartOnLine = Math.max(0, selectionStart - lineGlobalStart);
        const charsBeforeEndOnLine = Math.max(0, selectionEnd - lineGlobalStart);

        removedBeforeStart += Math.min(removeCount, charsBeforeStartOnLine);
        removedBeforeEnd += Math.min(removeCount, charsBeforeEndOnLine);

        lineGlobalStart += line.length + 1;
        return line.slice(removeCount);
      });

      const nextBlock = outdentedLines.join("\n");
      if (nextBlock === selectedBlock) {
        return;
      }

      const nextValue = `${value.slice(0, blockStart)}${nextBlock}${value.slice(blockEnd)}`;
      const nextSelectionStart = Math.max(blockStart, selectionStart - removedBeforeStart);
      const nextSelectionEnd = Math.max(nextSelectionStart, selectionEnd - removedBeforeEnd);
      commitJsonEditorUpdate(event, testIndex, field, nextValue, nextSelectionStart, nextSelectionEnd);
      return;
    }

    if (altKey || ctrlKey || metaKey) {
      return;
    }

    event.preventDefault();
    const currentLineStart = getLineStartIndex(value, selectionStart);
    const linePrefix = value.slice(currentLineStart, selectionStart);
    const baseIndentMatch = linePrefix.match(/^\s*/);
    const baseIndent = baseIndentMatch ? baseIndentMatch[0] : "";
    const previousChar = value.charAt(selectionStart - 1);
    const nextChar = value.charAt(selectionEnd);
    const opensBlock = previousChar === "{" || previousChar === "[";
    const closesBlock = nextChar === "}" || nextChar === "]";

    let insertedText = `\n${baseIndent}`;
    let nextCursor = selectionStart + insertedText.length;

    if (opensBlock && closesBlock) {
      insertedText = `\n${baseIndent}${JSON_EDITOR_INDENT}\n${baseIndent}`;
      nextCursor = selectionStart + 1 + baseIndent.length + JSON_EDITOR_INDENT.length;
    } else if (opensBlock) {
      insertedText = `\n${baseIndent}${JSON_EDITOR_INDENT}`;
      nextCursor = selectionStart + insertedText.length;
    }

    const nextValue = `${value.slice(0, selectionStart)}${insertedText}${value.slice(selectionEnd)}`;
    commitJsonEditorUpdate(event, testIndex, field, nextValue, nextCursor);
  }

  function handleResetCase(testIndex) {
    if (!entry) {
      return;
    }

    const originalCase = baselineCases[testIndex];
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
  const latestRunResults = Array.isArray(runState?.result?.results) ? runState.result.results : [];
  const hasLatestRun = Boolean(latestRunSummary || latestRunResults.length > 0);
  const runResultByTestId = latestRunResults.reduce((acc, result) => {
    const testId = String(result?.test_id || "");
    if (!testId) {
      return acc;
    }
    acc[testId] = result;
    return acc;
  }, {});
  const runOutcomeByTestId = latestRunResults.reduce((acc, result) => {
    const testId = String(result?.test_id || "");
    if (!testId) {
      return acc;
    }
    acc[testId] = normalizeRunOutcome(result?.outcome);
    return acc;
  }, {});
  const trimmedBaseUrl = baseUrlInput.trim();
  const isBaseUrlMissing = trimmedBaseUrl.length === 0;
  const trimmedRunBearerToken = runBearerToken.trim();
  const trimmedRunApiKeyValue = runApiKeyValue.trim();
  const trimmedRunApiKeyHeader = runApiKeyHeader.trim() || "X-API-Key";
  const isBearerTokenMissing = runAuthMode === "bearer" && trimmedRunBearerToken.length === 0;
  const isApiKeyMissing = runAuthMode === "api_key" && trimmedRunApiKeyValue.length === 0;
  const hasAuthInputError = isBearerTokenMissing || isApiKeyMissing;

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

    const runPayload = {
      api_title: entry?.title || entry?.filename || "Generated Test Suite",
      api_version: entry?.version || "Unknown",
      base_url: trimmedBaseUrl,
      test_cases: nextTestCases,
    };

    if (runAuthMode === "bearer") {
      runPayload.bearer_token = trimmedRunBearerToken;
    } else if (runAuthMode === "api_key") {
      runPayload.api_key = trimmedRunApiKeyValue;
      runPayload.api_key_header = trimmedRunApiKeyHeader;
    }

    return runPayload;
  }

  function handleRunTests() {
    if (!entry || !onRunTests || isBaseUrlMissing || hasAuthInputError) {
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
                      <div className="run-controls">
                        <button
                          type="button"
                          className="primary-button run-tests-button"
                          onClick={handleRunTests}
                          disabled={
                            generatedCases.length === 0
                            || hasJsonDraftErrors
                            || isBaseUrlMissing
                            || hasAuthInputError
                            || runState?.loading
                          }
                        >
                          {runState?.loading ? "Running..." : "Run Tests"}
                        </button>
                        <input
                          type="text"
                          className="base-url-input"
                          placeholder="https://api.example.com"
                          value={baseUrlInput}
                          onChange={(event) => setBaseUrlInput(event.target.value)}
                          aria-label="Base URL"
                        />
                      </div>
                      <div className="auth-controls">
                        <label className="auth-field">
                          <span className="auth-label">Run Auth</span>
                          <select
                            className="auth-select"
                            value={runAuthMode}
                            onChange={(event) => setRunAuthMode(event.target.value)}
                          >
                            <option value="none">None</option>
                            <option value="bearer">Bearer Token</option>
                            <option value="api_key">API Key Header</option>
                          </select>
                        </label>
                        {runAuthMode === "bearer" ? (
                          <label className="auth-field auth-field-wide">
                            <span className="auth-label">Bearer Token</span>
                            <input
                              type="password"
                              className="auth-input"
                              placeholder="ghp_..."
                              value={runBearerToken}
                              onChange={(event) => setRunBearerToken(event.target.value)}
                              autoComplete="off"
                            />
                          </label>
                        ) : null}
                        {runAuthMode === "api_key" ? (
                          <>
                            <label className="auth-field auth-field-wide">
                              <span className="auth-label">API Key Value</span>
                              <input
                                type="password"
                                className="auth-input"
                                placeholder="Enter API key"
                                value={runApiKeyValue}
                                onChange={(event) => setRunApiKeyValue(event.target.value)}
                                autoComplete="off"
                              />
                            </label>
                            <label className="auth-field">
                              <span className="auth-label">Header Name</span>
                              <input
                                type="text"
                                className="auth-input"
                                placeholder="X-API-Key"
                                value={runApiKeyHeader}
                                onChange={(event) => setRunApiKeyHeader(event.target.value)}
                                autoComplete="off"
                              />
                            </label>
                          </>
                        ) : null}
                      </div>
                      {isBaseUrlMissing ? <p className="json-error base-url-required">*Base URL required*.</p> : null}
                      {isBearerTokenMissing ? <p className="json-error">*Bearer token required*.</p> : null}
                      {isApiKeyMissing ? <p className="json-error">*API key value required*.</p> : null}
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
                        const categoryRunStats = categoryCases.reduce((acc, { testCase }) => {
                          const outcome = runOutcomeByTestId[testCase?.test_id || ""];
                          if (outcome === "PASS") {
                            acc.pass += 1;
                          } else if (outcome === "FAIL") {
                            acc.fail += 1;
                          } else if (outcome === "SKIP") {
                            acc.skip += 1;
                          }
                          return acc;
                        }, { pass: 0, fail: 0, skip: 0 });
                        return (
                          <details key={category} className="category-detail">
                            <summary onClick={handleSummaryToggleNoScroll}>
                              <span>{prettifyCategory(category)}</span>
                              <span className="category-summary-right">
                                <strong>{count}</strong>
                                {hasLatestRun ? (
                                  <span className="category-run-text">
                                    <span className="category-run-part category-run-part-pass">{categoryRunStats.pass}P</span>
                                    <span className="category-run-separator">/</span>
                                    <span className="category-run-part category-run-part-fail">{categoryRunStats.fail}F</span>
                                    <span className="category-run-separator">/</span>
                                    <span className="category-run-part category-run-part-skip">{categoryRunStats.skip}S</span>
                                  </span>
                                ) : null}
                              </span>
                            </summary>
                            <div className="category-body">
                              {categoryCases.length > 0 ? (
                                <div className="testcase-list">
                                  {categoryCases.map(({ testCase, index: testIndex }) => {
                                    const firstStep = testCase.steps?.[0] || null;
                                    const inputData = firstStep?.input_data ?? null;
                                    const expectedResult = testCase.expected_result ?? null;
                                    const originalCase = baselineCases[testIndex] || {};
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
                                    const runResult = runResultByTestId[testCase?.test_id || ""] || null;
                                    const testOutcome = runOutcomeByTestId[testCase?.test_id || ""];
                                    const outcomeClass = testOutcome ? `testcase-outcome-${testOutcome.toLowerCase()}` : "";
                                    const expectedStatusLabel = formatExpectedStatusLabel(
                                      runResult?.expected_status,
                                      runResult?.expected_status_any_of,
                                    );
                                    const actualStatusLabel = runResult?.actual_status ?? "No response";
                                    const runErrorMessage = String(runResult?.error_message || "").trim();
                                    const responseSnippet = String(runResult?.response_snippet || "").trim();
                                    const snippetMeta = parseResponseSnippetMeta(responseSnippet);
                                    const statusMatchesExpectation = isExpectedStatusMatch(
                                      runResult?.actual_status,
                                      runResult?.expected_status,
                                      runResult?.expected_status_any_of,
                                    );
                                    let fallbackReason = "Unexpected response from server.";
                                    if (!statusMatchesExpectation) {
                                      if (runResult?.actual_status === undefined || runResult?.actual_status === null) {
                                        fallbackReason = "Request failed before receiving a response.";
                                      } else {
                                        fallbackReason = `Expected ${expectedStatusLabel} but got ${actualStatusLabel}.`;
                                      }
                                    }
                                    const failureReasonRaw = runErrorMessage || snippetMeta.reason || fallbackReason;
                                    const failureReason = truncateText(failureReasonRaw, 220);
                                    const hasRawResponse = Boolean(snippetMeta.raw)
                                      && snippetMeta.raw !== failureReasonRaw
                                      && snippetMeta.raw !== failureReason;

                                    return (
                                      <details
                                        key={testCase.test_id || `${testCase.title}-${testIndex}`}
                                        className={`testcase-detail ${outcomeClass}`.trim()}
                                      >
                                        <summary onClick={handleSummaryToggleNoScroll}>
                                          <span>{testCase.test_id || "Test case"}</span>
                                          <span className="testcase-summary-right">
                                            <span>{testCase.method} {testCase.path}</span>
                                            {testOutcome ? (
                                              <span className={`result-pill result-pill-${testOutcome.toLowerCase()}`}>
                                                {testOutcome}
                                              </span>
                                            ) : null}
                                          </span>
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
                                              onKeyDown={(event) => handleJsonEditorKeyDown(testIndex, "input_data", event)}
                                              rows={8}
                                              spellCheck={false}
                                            />
                                            {inputDataError ? <p className="json-error">{inputDataError}</p> : null}
                                          </div>

                                          <div className="json-section">
                                            <span className="json-label">Expected Result</span>
                                            <textarea
                                              className={`json-editor json-editor-readonly ${testOutcome ? `json-editor-readonly-${testOutcome.toLowerCase()}` : ""}`.trim()}
                                              value={expectedResultText}
                                              rows={8}
                                              spellCheck={false}
                                              readOnly
                                            />
                                            {expectedResultError ? <p className="json-error">{expectedResultError}</p> : null}
                                          </div>

                                          {testOutcome === "FAIL" ? (
                                            <div className="run-feedback run-feedback-fail">
                                              <p className="run-feedback-title">Failure Details</p>
                                              <p className="run-feedback-line"><strong>Expected status:</strong> {expectedStatusLabel}</p>
                                              <p className="run-feedback-line"><strong>Actual status:</strong> {actualStatusLabel}</p>
                                              <p className="run-feedback-line"><strong>Reason:</strong> {failureReason}</p>
                                              {snippetMeta.documentationUrl ? (
                                                <p className="run-feedback-line">
                                                  <strong>Docs:</strong>{" "}
                                                  <a
                                                    href={snippetMeta.documentationUrl}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="run-feedback-link"
                                                  >
                                                    {snippetMeta.documentationUrl}
                                                  </a>
                                                </p>
                                              ) : null}
                                              {hasRawResponse ? (
                                                <details className="run-feedback-details">
                                                  <summary>Raw response</summary>
                                                  <pre className="run-feedback-pre">{snippetMeta.raw}</pre>
                                                </details>
                                              ) : null}
                                            </div>
                                          ) : null}

                                          {testOutcome === "SKIP" ? (
                                            <div className="run-feedback run-feedback-skip">
                                              <p className="run-feedback-title">Skipped</p>
                                              <p className="run-feedback-line">{runErrorMessage || "Runner skipped this test."}</p>
                                            </div>
                                          ) : null}
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
  const [theme, setTheme] = useState(() => loadTheme());
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
  const isDarkTheme = theme === "dark";

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
    saveTheme(theme);
  }, [theme]);

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
      return { loading: false, error: "", result: null, baselineTests: null };
    }
    return testRunBySpecId[selectedEntry.id] || { loading: false, error: "", result: null, baselineTests: null };
  }, [selectedEntry, testRunBySpecId]);

  function setRunBaselineForSpec(specId, executedTests) {
    const normalizedTests = Array.isArray(executedTests) ? (cloneJsonValue(executedTests) || []) : [];

    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }

        const nextTests = cloneJsonValue(normalizedTests) || [];
        return {
          ...entry,
          generatedTests: nextTests,
          originalGeneratedTests: cloneJsonValue(nextTests) || [],
          generatedSuite: entry.generatedSuite
            ? { ...entry.generatedSuite, test_cases: nextTests }
            : entry.generatedSuite,
        };
      }),
    );

    if (!session?.userId) {
      return;
    }

    setSpecCache((current) => {
      const userCache = { ...(current[session.userId] || {}) };
      const existing = userCache[specId];
      if (!existing) {
        return current;
      }

      const nextTests = cloneJsonValue(normalizedTests) || [];
      userCache[specId] = {
        ...existing,
        generatedTests: nextTests,
        originalGeneratedTests: cloneJsonValue(nextTests) || [],
        generatedSuite: existing.generatedSuite
          ? { ...existing.generatedSuite, test_cases: nextTests }
          : existing.generatedSuite,
      };

      return {
        ...current,
        [session.userId]: userCache,
      };
    });
  }

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
    const runBaseline = Array.isArray(testRunBySpecId[specId]?.baselineTests)
      ? testRunBySpecId[specId].baselineTests
      : null;

    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }

        const tests = Array.isArray(entry.generatedTests) ? entry.generatedTests : [];
        const originalTests = Array.isArray(runBaseline) && runBaseline.length > 0
          ? runBaseline
          : (Array.isArray(entry.originalGeneratedTests) ? entry.originalGeneratedTests : tests);
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
      const originalTests = Array.isArray(runBaseline) && runBaseline.length > 0
        ? runBaseline
        : (Array.isArray(existing.originalGeneratedTests) ? existing.originalGeneratedTests : currentTests);
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
        baselineTests: Array.isArray(current[specId]?.baselineTests) ? current[specId].baselineTests : null,
      },
    }));

    try {
      const payload = await runGeneratedTests(session.token, suitePayload);
      const executedTests = Array.isArray(suitePayload?.test_cases) ? suitePayload.test_cases : [];
      setRunBaselineForSpec(specId, executedTests);
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          loading: false,
          error: "",
          result: payload || null,
          baselineTests: cloneJsonValue(executedTests) || [],
        },
      }));
    } catch (error) {
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          loading: false,
          error: error.message,
          result: current[specId]?.result || null,
          baselineTests: Array.isArray(current[specId]?.baselineTests) ? current[specId].baselineTests : null,
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

  function handleThemeToggle() {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }

  return (
    <div className="app-shell" data-theme={theme}>
      <div className="background-orb orb-one" />
      <div className="background-orb orb-two" />

      <header className="hero">
        <div className="hero-main">
          <p className="eyebrow">API Contract Testing</p>
          <h1>ContractGuard</h1>
          <p className="hero-copy">
            Service status: <span>{apiStatus}</span>
          </p>
        </div>
        <div className="hero-aside">
          <button
            type="button"
            className="theme-toggle"
            onClick={handleThemeToggle}
            aria-pressed={isDarkTheme}
            aria-label={isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}
          >
            <span className="theme-toggle-icon" aria-hidden="true">
              {isDarkTheme ? (
                <svg viewBox="0 0 24 24" role="presentation" focusable="false">
                  <circle cx="12" cy="12" r="4.2" />
                  <line x1="12" y1="1.6" x2="12" y2="5.1" />
                  <line x1="12" y1="18.9" x2="12" y2="22.4" />
                  <line x1="1.6" y1="12" x2="5.1" y2="12" />
                  <line x1="18.9" y1="12" x2="22.4" y2="12" />
                  <line x1="4.2" y1="4.2" x2="6.8" y2="6.8" />
                  <line x1="17.2" y1="17.2" x2="19.8" y2="19.8" />
                  <line x1="17.2" y1="6.8" x2="19.8" y2="4.2" />
                  <line x1="4.2" y1="19.8" x2="6.8" y2="17.2" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" role="presentation" focusable="false">
                  <path d="M15.8 2.9a9.6 9.6 0 1 0 5.3 16.9 9.2 9.2 0 1 1-5.3-16.9z" />
                </svg>
              )}
            </span>
            <span className="sr-only">{isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}</span>
          </button>
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
        </div>
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
