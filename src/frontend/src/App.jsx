import { useEffect, useMemo, useRef, useState } from "react";
import {
  addLlmModel,
  API_BASE_URL,
  clearLogisticsRuns,
  clearSpecs,
  deleteLogisticsRun,
  deleteLlmModel,
  fetchLogisticsRunDetail,
  fetchLatestRunForSpec,
  fetchLlmProviderModels,
  fetchLlmSettings,
  healthCheck,
  listLogisticsRuns,
  listSpecs,
  loginUser,
  requestLlmFailureAnalysis,
  requestSpecGroundedChat,
  registerUser,
  runGeneratedTests,
  updateLlmSettings,
  uploadSpecFile,
} from "./api.js";
import { buildSpecPreview } from "./testPreview.js";

const SESSION_KEY = "contractguard.session.v1";
const SPEC_CACHE_KEY = "contractguard.spec-cache.v1";
const THEME_KEY = "contractguard.theme.v1";
const CHAT_THREAD_KEY = "contractguard.chat-thread.v1";
const CHAT_CONTEXT_SPEC = "spec";
const SETTINGS_TAB_MODEL = "model";
const SETTINGS_TAB_CUSTOM = "custom";
const SETTINGS_MODEL_PROVIDERS = Object.freeze([
  { value: "ollama", label: "Ollama" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic (Claude)" },
]);
const SPEC_FILE_EXTENSIONS = [".json", ".yaml", ".yml"];
const JSON_EDITOR_INDENT = "  ";
const FAILURE_REASON_KEYS = ["message", "detail", "error", "reason", "title", "description"];
const CHAT_MESSAGE_ROLES = new Set(["user", "assistant", "system"]);
const DEFAULT_CHAT_RUNTIME_CONFIG = Object.freeze({
  modelId: "qwen3-coder:latest",
  userInstruction: "",
});
const DEFAULT_LLM_SETTINGS = Object.freeze({
  active_model_id: "",
  default_model_id: "",
  custom_instruction: "",
  models: [],
});
const DEFAULT_ADD_MODEL_FORM = Object.freeze({
  provider: "openai",
  model: "",
  label: "",
  base_url: "",
  api_key: "",
});
const WORKSPACE_VIEW_DASHBOARD = "dashboard";
const WORKSPACE_VIEW_LOGS = "logs";
const LOGISTICS_FILTER_STATES = new Set(["all", "passed", "failed"]);
const DEFAULT_LOGISTICS_FILTERS = Object.freeze({
  specQuery: "",
  state: "all",
});
const LOGISTICS_PAGE_LIMIT = 25;

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
    return parsed && typeof parsed === "object" ? compactSpecCache(parsed) : {};
  } catch {
    return {};
  }
}

function saveSpecCache(cache) {
  try {
    localStorage.setItem(SPEC_CACHE_KEY, JSON.stringify(compactSpecCache(cache)));
  } catch {
    try {
      localStorage.removeItem(SPEC_CACHE_KEY);
    } catch {
      // Ignore storage write/remove failures.
    }
  }
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

function getDefaultChatContext() {
  return {
    mode: CHAT_CONTEXT_SPEC,
    specId: null,
  };
}

function createChatMessageId(prefix = "chat") {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeChatContext(context) {
  if (!context || typeof context !== "object") {
    return getDefaultChatContext();
  }

  const mode = context.mode === CHAT_CONTEXT_SPEC ? CHAT_CONTEXT_SPEC : null;
  if (!mode) {
    return getDefaultChatContext();
  }

  const specIdNumber = Number(context.specId);
  if (!Number.isFinite(specIdNumber)) {
    return getDefaultChatContext();
  }

  return {
    mode: CHAT_CONTEXT_SPEC,
    specId: specIdNumber,
  };
}

function normalizeChatRuntimeConfig(runtimeConfig) {
  return {
    modelId: String(runtimeConfig?.modelId || DEFAULT_CHAT_RUNTIME_CONFIG.modelId),
    userInstruction: String(runtimeConfig?.userInstruction || ""),
  };
}

function normalizeLlmModelEntry(entry, index = 0) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const id = String(entry.id || `model-${index}`);
  const provider = String(entry.provider || "").trim().toLowerCase();
  const model = String(entry.model || "").trim();
  if (!id || !provider || !model) {
    return null;
  }
  const label = String(entry.label || "").trim() || model;
  const source = String(entry.source || "user").trim().toLowerCase();
  const baseUrl = typeof entry.base_url === "string" ? entry.base_url : null;
  const hasApiKey = Boolean(entry.has_api_key);
  const apiKeyMasked = String(entry.api_key_masked || "");
  return {
    id,
    provider,
    model,
    label,
    source,
    base_url: baseUrl,
    has_api_key: hasApiKey,
    api_key_masked: apiKeyMasked,
  };
}

function normalizeProviderModelEntry(entry, index = 0) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const id = String(entry.id || `provider-model-${index}`).trim();
  if (!id) {
    return null;
  }
  const label = String(entry.label || "").trim() || id;
  return { id, label };
}

function normalizeProviderModelCatalog(models) {
  return Array.isArray(models)
    ? models
      .map((entry, index) => normalizeProviderModelEntry(entry, index))
      .filter(Boolean)
    : [];
}

function normalizeLlmSettings(settings) {
  const models = Array.isArray(settings?.models)
    ? settings.models
      .map((entry, index) => normalizeLlmModelEntry(entry, index))
      .filter(Boolean)
    : [];
  const activeModelId = String(settings?.active_model_id || "");
  const defaultModelId = String(settings?.default_model_id || "");
  const activeExists = models.some((entry) => entry.id === activeModelId);
  const defaultExists = models.some((entry) => entry.id === defaultModelId);
  const fallbackId = defaultExists ? defaultModelId : (models[0]?.id || "");
  return {
    active_model_id: activeExists ? activeModelId : fallbackId,
    default_model_id: defaultExists ? defaultModelId : fallbackId,
    custom_instruction: String(settings?.custom_instruction || ""),
    models,
  };
}

function getActiveLlmModelEntry(settings) {
  const activeModelId = String(settings?.active_model_id || "");
  const models = Array.isArray(settings?.models) ? settings.models : [];
  return models.find((entry) => String(entry?.id || "") === activeModelId) || models[0] || null;
}

function normalizeChatMessage(message, index = 0) {
  if (!message || typeof message !== "object") {
    return null;
  }

  const content = String(message.content || "").trim();
  if (!content) {
    return null;
  }

  const role = CHAT_MESSAGE_ROLES.has(message.role) ? message.role : "system";
  const timestampRaw = String(message.createdAt || "").trim();
  const timestamp = Date.parse(timestampRaw);
  const createdAt = Number.isNaN(timestamp) ? new Date().toISOString() : new Date(timestamp).toISOString();
  const id = String(message.id || createChatMessageId(`msg-${role}-${index}`));

  return {
    id,
    role,
    content,
    createdAt,
    context: normalizeChatContext(message.context),
    meta: message.meta && typeof message.meta === "object" ? message.meta : {},
  };
}

function normalizeChatThread(thread) {
  const normalizedMessages = Array.isArray(thread?.messages)
    ? thread.messages
      .map((message, index) => normalizeChatMessage(message, index))
      .filter(Boolean)
    : [];

  return {
    messages: normalizedMessages,
    activeContext: normalizeChatContext(thread?.activeContext),
    runtimeConfig: normalizeChatRuntimeConfig(thread?.runtimeConfig),
  };
}

function loadChatThread(userId) {
  if (!userId) {
    return normalizeChatThread(null);
  }

  try {
    const raw = localStorage.getItem(CHAT_THREAD_KEY);
    if (!raw) {
      return normalizeChatThread(null);
    }

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return normalizeChatThread(null);
    }

    return normalizeChatThread(parsed[userId]);
  } catch {
    return normalizeChatThread(null);
  }
}

function saveChatThread(userId, thread) {
  if (!userId) {
    return;
  }

  try {
    const normalizedThread = normalizeChatThread(thread);
    const existingRaw = localStorage.getItem(CHAT_THREAD_KEY);
    const existingParsed = existingRaw ? JSON.parse(existingRaw) : {};
    const safeStore = existingParsed && typeof existingParsed === "object" ? existingParsed : {};
    safeStore[userId] = {
      ...normalizedThread,
      updatedAt: new Date().toISOString(),
    };
    localStorage.setItem(CHAT_THREAD_KEY, JSON.stringify(safeStore));
  } catch {
    // Ignore storage write errors.
  }
}

function createChatMessage({ role, content, context, meta }) {
  return {
    id: createChatMessageId(role || "chat"),
    role: CHAT_MESSAGE_ROLES.has(role) ? role : "system",
    content: String(content || "").trim(),
    createdAt: new Date().toISOString(),
    context: normalizeChatContext(context),
    meta: meta && typeof meta === "object" ? meta : {},
  };
}

function formatChatTime(value) {
  const parsed = Date.parse(String(value || ""));
  if (Number.isNaN(parsed)) {
    return "";
  }
  return new Date(parsed).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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

function formatDuration(durationMs) {
  const totalMs = toSafeCount(durationMs);
  if (totalMs <= 0) {
    return "0 ms";
  }
  if (totalMs < 1000) {
    return `${totalMs} ms`;
  }
  const totalSeconds = totalMs / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(totalSeconds >= 10 ? 1 : 2)} s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

function normalizeLogisticsState(value) {
  const normalized = String(value || "all").trim().toLowerCase() || "all";
  return LOGISTICS_FILTER_STATES.has(normalized) ? normalized : "all";
}

function normalizeLogisticsFilters(filters) {
  return {
    specQuery: String(filters?.specQuery || "").trim(),
    state: normalizeLogisticsState(filters?.state),
  };
}

function normalizeLogisticsRunCard(entry, index = 0) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const spec = entry.spec && typeof entry.spec === "object" ? entry.spec : {};
  const run = entry.run && typeof entry.run === "object" ? entry.run : {};
  const auth = entry.auth && typeof entry.auth === "object" ? entry.auth : {};
  const ai = entry.ai && typeof entry.ai === "object" ? entry.ai : {};
  const runId = Number(run.id);
  if (!Number.isFinite(runId)) {
    return null;
  }
  const summary = normalizeOutcomeSummary(run.summary);
  const modelList = Array.isArray(ai.model_list)
    ? ai.model_list.map((model) => String(model || "").trim()).filter(Boolean)
    : [];

  return {
    id: runId,
    spec: {
      id: Number(spec.id),
      title: String(spec.title || spec.filename || `Spec ${index + 1}`),
      filename: String(spec.filename || ""),
      version: String(spec.version || "Unknown"),
      uploadedAt: spec.uploaded_at || null,
    },
    run: {
      id: runId,
      createdAt: run.created_at || null,
      summary,
      durationMs: toSafeCount(run.duration_ms),
      baseUrl: String(run.base_url || ""),
    },
    auth: {
      provided: Boolean(auth.provided),
      mode: String(auth.mode || "none"),
    },
    ai: {
      hasAny: Boolean(ai.has_any),
      hasExplanations: Boolean(ai.has_explanations),
      hasSuggestions: Boolean(ai.has_suggestions),
      explanationCount: toSafeCount(ai.explanation_count),
      suggestionCount: toSafeCount(ai.suggestion_count),
      modelList,
    },
  };
}

function normalizeLogisticsRunCards(items) {
  return Array.isArray(items)
    ? items.map((entry, index) => normalizeLogisticsRunCard(entry, index)).filter(Boolean)
    : [];
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

function buildUniqueTestId(existingTests, preferredId) {
  const existingIds = new Set(
    (Array.isArray(existingTests) ? existingTests : [])
      .map((testCase) => String(testCase?.test_id || "").trim())
      .filter(Boolean),
  );
  const base = String(preferredId || "TC-LLM-FOLLOWUP").trim() || "TC-LLM-FOLLOWUP";
  if (!existingIds.has(base)) {
    return base;
  }
  let index = 1;
  while (true) {
    const candidate = `${base}-ADDED-${String(index).padStart(2, "0")}`;
    if (!existingIds.has(candidate)) {
      return candidate;
    }
    index += 1;
  }
}

function normalizeSuggestedTestCaseForSuite(suggestedTestCase, existingTests, originCategory = "") {
  if (!suggestedTestCase || typeof suggestedTestCase !== "object" || Array.isArray(suggestedTestCase)) {
    return null;
  }
  const copy = cloneJsonValue(suggestedTestCase) || {};
  const method = String(copy.method || "GET").toUpperCase();
  const path = String(copy.path || "/");
  const steps = Array.isArray(copy.steps) && copy.steps.length > 0 ? copy.steps : [{ step_number: 1, action: "Execute request", input_data: {} }];
  const firstStep = steps[0] && typeof steps[0] === "object"
    ? { ...steps[0], input_data: (steps[0].input_data && typeof steps[0].input_data === "object") ? steps[0].input_data : {} }
    : { step_number: 1, action: "Execute request", input_data: {} };
  const expectedResult = copy.expected_result && typeof copy.expected_result === "object"
    ? copy.expected_result
    : { status_code: 200, description: "LLM suggested follow-up case." };

  return {
    test_id: buildUniqueTestId(existingTests, copy.test_id),
    title: String(copy.title || `LLM follow-up for ${method} ${path}`),
    category: String(originCategory || copy.category || "llm_followup"),
    requirement_ref: String(copy.requirement_ref || "llm_assistant"),
    method,
    path,
    priority: String(copy.priority || "medium"),
    preconditions: Array.isArray(copy.preconditions) ? copy.preconditions.map((item) => String(item)) : [],
    steps: [firstStep],
    expected_result: expectedResult,
  };
}

function buildSuggestedCaseSignature(testCase) {
  if (!testCase || typeof testCase !== "object" || Array.isArray(testCase)) {
    return "";
  }
  const firstStep = Array.isArray(testCase.steps) && testCase.steps.length > 0 && testCase.steps[0]
    ? testCase.steps[0]
    : {};
  const normalized = {
    title: String(testCase.title || ""),
    category: String(testCase.category || ""),
    requirement_ref: String(testCase.requirement_ref || ""),
    method: String(testCase.method || "").toUpperCase(),
    path: String(testCase.path || ""),
    priority: String(testCase.priority || ""),
    preconditions: Array.isArray(testCase.preconditions) ? testCase.preconditions.map((item) => String(item)) : [],
    step: {
      action: String(firstStep?.action || ""),
      input_data: firstStep?.input_data && typeof firstStep.input_data === "object" ? firstStep.input_data : {},
    },
    expected_result: testCase.expected_result && typeof testCase.expected_result === "object"
      ? testCase.expected_result
      : {},
  };
  return JSON.stringify(normalized);
}

function hasSuggestedCaseAlreadyBeenAdded(existingTests, candidateCase) {
  const candidateSignature = buildSuggestedCaseSignature(candidateCase);
  if (!candidateSignature) {
    return false;
  }
  return (Array.isArray(existingTests) ? existingTests : []).some(
    (testCase) => buildSuggestedCaseSignature(testCase) === candidateSignature,
  );
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

const EXPLANATION_LEADING_LABEL_RE = /\b(?:Confirmed|Likely)\s+cause\s*:/gi;
const EXPLANATION_SECTION_BREAK_RE = /\b(?:Why\s+likely|Check\s+next)\s*:/gi;
const EXPLANATION_ANY_LABEL_RE = /\b(?:Confirmed|Likely)\s+cause\s*:|\b(?:Why\s+likely|Check\s+next)\s*:/i;

function ensureSentenceTerminalPunctuation(text) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (!compact) {
    return "";
  }
  return /[.!?]$/.test(compact) ? compact : `${compact}.`;
}

function formatExplanationForDisplay(explanationText) {
  const raw = String(explanationText || "").trim();
  if (!raw) {
    return "";
  }

  const normalized = raw.replace(/\s+/g, " ").trim();
  if (!EXPLANATION_ANY_LABEL_RE.test(normalized)) {
    return normalized;
  }

  const withoutLeadingLabel = normalized.replace(EXPLANATION_LEADING_LABEL_RE, "");
  const sectionParts = withoutLeadingLabel
    .replace(EXPLANATION_SECTION_BREAK_RE, " ||| ")
    .split("|||")
    .map((part) => ensureSentenceTerminalPunctuation(part))
    .filter(Boolean);

  if (sectionParts.length > 0) {
    return sectionParts.join(" ");
  }

  return withoutLeadingLabel;
}

function getEmptyRunState() {
  return {
    loading: false,
    error: "",
    result: null,
    baselineTests: null,
    runId: null,
    llmByTestId: {},
  };
}

function toSafeCount(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, Math.trunc(parsed));
}

function normalizeOutcomeSummary(summary) {
  const source = summary && typeof summary === "object" ? summary : {};
  const passed = toSafeCount(source.passed);
  const failed = toSafeCount(source.failed);
  const skipped = toSafeCount(source.skipped);
  const providedTotal = toSafeCount(source.total);
  const computedTotal = passed + failed + skipped;
  const total = Math.max(providedTotal, computedTotal);
  return {
    total,
    passed,
    failed,
    skipped,
  };
}

function summarizeRunResults(results) {
  const runResults = Array.isArray(results) ? results : [];
  const summary = {
    total: 0,
    passed: 0,
    failed: 0,
    skipped: 0,
  };
  for (const result of runResults) {
    const outcome = normalizeRunOutcome(result?.outcome);
    summary.total += 1;
    if (outcome === "PASS") {
      summary.passed += 1;
    } else if (outcome === "FAIL") {
      summary.failed += 1;
    } else if (outcome === "SKIP") {
      summary.skipped += 1;
    }
  }
  return summary;
}

function resolveRunSummary(summary, results) {
  const normalizedSummary = normalizeOutcomeSummary(summary);
  if (normalizedSummary.total > 0) {
    return normalizedSummary;
  }
  const derivedSummary = summarizeRunResults(results);
  return derivedSummary.total > 0 ? derivedSummary : normalizedSummary;
}

function buildLatestRunState(latestRun, fallbackEntry = null) {
  const source = latestRun && typeof latestRun === "object" ? latestRun : {};
  const fallback = fallbackEntry && typeof fallbackEntry === "object" ? fallbackEntry : {};

  const sourceId = Number(source.id ?? source.run_id);
  const fallbackId = Number(fallback.latestRunId);
  const latestRunId = Number.isFinite(sourceId)
    ? sourceId
    : (Number.isFinite(fallbackId) ? fallbackId : null);

  const createdAt = source.created_at || source.createdAt || fallback.latestRunCreatedAt || null;
  const sourceSummary = source.summary && typeof source.summary === "object"
    ? source.summary
    : (fallback.latestRunSummary && typeof fallback.latestRunSummary === "object"
      ? fallback.latestRunSummary
      : null);
  const latestRunSummary = sourceSummary || latestRunId !== null
    ? normalizeOutcomeSummary(sourceSummary)
    : null;

  return {
    latestRunId,
    latestRunCreatedAt: createdAt || null,
    latestRunSummary,
  };
}

function aggregateWorkspaceLatestRunSummary(entries) {
  const sourceEntries = Array.isArray(entries) ? entries : [];
  const totals = {
    total: 0,
    passed: 0,
    failed: 0,
    skipped: 0,
    specsWithRuns: 0,
  };
  for (const entry of sourceEntries) {
    const runId = Number(entry?.latestRunId);
    const hasRun = Number.isFinite(runId);
    if (!hasRun) {
      continue;
    }
    totals.specsWithRuns += 1;
    const summary = normalizeOutcomeSummary(entry?.latestRunSummary);
    totals.total += summary.total;
    totals.passed += summary.passed;
    totals.failed += summary.failed;
    totals.skipped += summary.skipped;
  }
  return totals;
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

function normalizeRunConfig(config, fallback = null) {
  const base = fallback && typeof fallback === "object"
    ? fallback
    : {
        baseUrl: "",
        authMode: "none",
        bearerToken: "",
        apiKeyValue: "",
        apiKeyHeader: "X-API-Key",
      };
  const source = config && typeof config === "object" ? config : {};
  const authModeRaw = String(source.authMode ?? base.authMode ?? "none");
  const authMode = authModeRaw === "bearer" || authModeRaw === "api_key" ? authModeRaw : "none";
  const apiKeyHeader = String(source.apiKeyHeader ?? base.apiKeyHeader ?? "X-API-Key").trim() || "X-API-Key";

  return {
    baseUrl: String(source.baseUrl ?? base.baseUrl ?? ""),
    authMode,
    bearerToken: String(source.bearerToken ?? base.bearerToken ?? ""),
    apiKeyValue: String(source.apiKeyValue ?? base.apiKeyValue ?? ""),
    apiKeyHeader,
  };
}

function buildDefaultRunConfig(entry) {
  return normalizeRunConfig({
    baseUrl: getDefaultBaseUrl(entry),
    authMode: "none",
    bearerToken: "",
    apiKeyValue: "",
    apiKeyHeader: "X-API-Key",
  });
}

function resolveRunConfigs(entry, cachedEntry = null) {
  const defaultConfig = normalizeRunConfig(cachedEntry?.runConfigDefault, buildDefaultRunConfig(entry));
  const currentConfig = normalizeRunConfig(cachedEntry?.runConfigCurrent, defaultConfig);
  return {
    runConfigDefault: defaultConfig,
    runConfigCurrent: currentConfig,
  };
}

function resolveUploadDefaultTests(entry, cachedEntry = null) {
  const cachedDefaults = Array.isArray(cachedEntry?.uploadDefaultTests) ? cachedEntry.uploadDefaultTests : null;
  if (cachedDefaults && cachedDefaults.length > 0) {
    return cloneJsonValue(cachedDefaults) || [];
  }
  const cachedOriginal = Array.isArray(cachedEntry?.originalGeneratedTests) ? cachedEntry.originalGeneratedTests : null;
  if (cachedOriginal && cachedOriginal.length > 0) {
    return cloneJsonValue(cachedOriginal) || [];
  }
  const cachedGenerated = Array.isArray(cachedEntry?.generatedTests) ? cachedEntry.generatedTests : null;
  if (cachedGenerated && cachedGenerated.length > 0) {
    return cloneJsonValue(cachedGenerated) || [];
  }
  const suiteCases = Array.isArray(entry?.generatedSuite?.test_cases) ? entry.generatedSuite.test_cases : [];
  return cloneJsonValue(suiteCases) || [];
}

function buildSpecCacheEntry(entry, fallbackEntry = null) {
  const source = entry && typeof entry === "object" ? entry : {};
  const fallback = fallbackEntry && typeof fallbackEntry === "object" ? fallbackEntry : {};

  const generatedTests = Array.isArray(source.generatedTests)
    ? source.generatedTests
    : (Array.isArray(fallback.generatedTests) ? fallback.generatedTests : []);
  const uploadDefaultTests = Array.isArray(source.uploadDefaultTests)
    ? source.uploadDefaultTests
    : (
        Array.isArray(source.originalGeneratedTests)
          ? source.originalGeneratedTests
          : (
              Array.isArray(fallback.uploadDefaultTests)
                ? fallback.uploadDefaultTests
                : generatedTests
            )
      );
  const runConfigDefault = normalizeRunConfig(
    source.runConfigDefault ?? fallback.runConfigDefault,
    buildDefaultRunConfig(source),
  );
  const runConfigCurrent = normalizeRunConfig(
    source.runConfigCurrent ?? fallback.runConfigCurrent,
    runConfigDefault,
  );

  return {
    filename: String(source.filename ?? fallback.filename ?? ""),
    uploadedAt: source.uploadedAt ?? source.created_at ?? source.createdAt ?? fallback.uploadedAt ?? fallback.createdAt ?? null,
    createdAt: source.createdAt ?? source.created_at ?? source.uploadedAt ?? fallback.createdAt ?? fallback.uploadedAt ?? null,
    preview: source.preview && typeof source.preview === "object"
      ? source.preview
      : (fallback.preview && typeof fallback.preview === "object" ? fallback.preview : null),
    generatedTests: Array.isArray(generatedTests) ? generatedTests : [],
    uploadDefaultTests: Array.isArray(uploadDefaultTests) ? uploadDefaultTests : [],
    runConfigDefault,
    runConfigCurrent,
  };
}

function compactSpecCache(cache) {
  if (!cache || typeof cache !== "object") {
    return {};
  }

  const next = {};
  for (const [userId, value] of Object.entries(cache)) {
    if (!value || typeof value !== "object") {
      continue;
    }
    const nextUserCache = {};
    for (const [specId, entry] of Object.entries(value)) {
      if (!entry || typeof entry !== "object") {
        continue;
      }
      nextUserCache[specId] = buildSpecCacheEntry(entry);
    }
    next[userId] = nextUserCache;
  }

  return next;
}

function StatCard({ label, value, accent }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className={`stat-value ${accent}`}>{value}</strong>
    </div>
  );
}

function OutcomeDonutCard({
  title,
  summary,
  hasData,
  emptyLabel = "No run yet",
  totalLabel = "Total tests",
}) {
  const normalized = normalizeOutcomeSummary(summary);
  const total = normalized.total;
  const passed = normalized.passed;
  const failed = normalized.failed;
  const skipped = normalized.skipped;
  const passDegrees = total > 0 ? (passed / total) * 360 : 0;
  const failDegrees = total > 0 ? (failed / total) * 360 : 0;
  const skipDegrees = Math.max(0, 360 - passDegrees - failDegrees);
  const donutStyle = hasData && total > 0
    ? {
        background: `conic-gradient(
          var(--outcome-pass) 0deg ${passDegrees}deg,
          var(--outcome-fail) ${passDegrees}deg ${passDegrees + failDegrees}deg,
          var(--outcome-skip) ${passDegrees + failDegrees}deg ${passDegrees + failDegrees + skipDegrees}deg
        )`,
      }
    : {};

  return (
    <div className={`outcome-card ${hasData ? "" : "outcome-card-empty"}`.trim()}>
      <div className="outcome-card-header">
        <h3>{title}</h3>
      </div>
      <div className="outcome-card-body">
        <div className="outcome-donut" style={donutStyle} aria-hidden="true">
          <div className="outcome-donut-center">
            <strong>{hasData ? total : "--"}</strong>
            <span>{hasData ? totalLabel : emptyLabel}</span>
          </div>
        </div>
        <div className="outcome-legend">
          <div className="outcome-legend-row">
            <span className="outcome-dot outcome-dot-pass" />
            <span>Passed</span>
            <strong>{hasData ? passed : "--"}</strong>
          </div>
          <div className="outcome-legend-row">
            <span className="outcome-dot outcome-dot-fail" />
            <span>Failed</span>
            <strong>{hasData ? failed : "--"}</strong>
          </div>
          <div className="outcome-legend-row">
            <span className="outcome-dot outcome-dot-skip" />
            <span>Skipped</span>
            <strong>{hasData ? skipped : "--"}</strong>
          </div>
        </div>
      </div>
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
      <div className="panel-header upload-panel-header">
        <p className="eyebrow">Uploads</p>
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

function LogisticsPanel({
  rows,
  loading,
  loadingMore,
  error,
  message,
  filterDraft,
  onFilterDraftChange,
  hasMore,
  onLoadMore,
  onOpenDetail,
  onDeleteRun,
  deletingRunId,
  onClearAll,
  clearLoading,
}) {
  return (
    <section className="panel logistics-panel">
      <div className="panel-header">
        <div className="panel-header-row">
          <div>
            <p className="eyebrow">Workspace</p>
            <h2>Logistics</h2>
          </div>
          <button
            type="button"
            className="danger-button"
            onClick={onClearAll}
            disabled={clearLoading || (loading && rows.length === 0)}
          >
            {clearLoading ? "Clearing..." : "Clear Logs"}
          </button>
        </div>
        <p className="muted">Run logs only. Uploaded specs remain after clearing logs.</p>
      </div>

      <div className="logistics-filter-bar">
        <label className="field logistics-filter-field">
          <span>Spec Search</span>
          <input
            type="text"
            value={String(filterDraft?.specQuery || "")}
            onChange={(event) => onFilterDraftChange?.({ specQuery: event.target.value })}
            placeholder="Search by title, file, or version"
          />
        </label>
        <label className="field logistics-filter-field logistics-filter-field-state">
          <span>Filter</span>
          <select
            value={normalizeLogisticsState(filterDraft?.state)}
            onChange={(event) => onFilterDraftChange?.({ state: event.target.value })}
          >
            <option value="all">All Runs</option>
            <option value="passed">Passed</option>
            <option value="failed">Failed</option>
          </select>
        </label>
      </div>

      {message ? <p className="message success">{message}</p> : null}
      {error ? <p className="message error">{error}</p> : null}

      {loading && rows.length === 0 ? (
        <div className="empty-state">
          <p>Loading run logs...</p>
          <span>Pulling compact run cards for this workspace.</span>
        </div>
      ) : null}

      {!loading && rows.length === 0 ? (
        <div className="empty-state">
          <p>No run logs found.</p>
          <span>Logs are created after a test run is saved.</span>
        </div>
      ) : null}

      {rows.length > 0 ? (
        <>
          <div className="logistics-list">
            {rows.map((row) => (
              <article key={row.id} className="logistics-row">
                <button type="button" className="logistics-row-main" onClick={() => onOpenDetail?.(row.id)}>
                  <div className="logistics-row-top">
                    <strong>{row.spec.title}</strong>
                  </div>
                  <div className="logistics-kv-grid">
                    <div className="logistics-kv-item">
                      <span className="logistics-kv-label">Spec ID</span>
                      <strong>{Number.isFinite(row.spec.id) ? `#${row.spec.id}` : "-"}</strong>
                    </div>
                    <div className="logistics-kv-item">
                      <span className="logistics-kv-label">Version</span>
                      <strong>{row.spec.version || "Unknown"}</strong>
                    </div>
                    <div className="logistics-kv-item">
                      <span className="logistics-kv-label">Duration</span>
                      <strong>{formatDuration(row.run.durationMs)}</strong>
                    </div>
                    <div className="logistics-kv-item logistics-kv-item-wide">
                      <span className="logistics-kv-label">Base URL</span>
                      <strong className="logistics-url-value">{row.run.baseUrl || "No base URL"}</strong>
                    </div>
                  </div>
                  <div className="logistics-row-outcomes">
                    <span className="run-chip run-chip-total">Total: {row.run.summary.total}</span>
                    <span className="run-chip run-chip-pass">Passed: {row.run.summary.passed}</span>
                    <span className="run-chip run-chip-fail">Failed: {row.run.summary.failed}</span>
                    <span className="run-chip run-chip-skip">Skipped: {row.run.summary.skipped}</span>
                  </div>
                  <div className="logistics-row-bottom">
                    <div className="logistics-row-meta">
                      <span>
                        Auth: {row.auth.provided ? `Provided (${String(row.auth.mode || "none").replaceAll("_", " ")})` : "Not provided"}
                      </span>
                      <span>
                        AI: {row.ai.hasAny
                          ? `${row.ai.explanationCount} explanation${row.ai.explanationCount === 1 ? "" : "s"}, ${row.ai.suggestionCount} suggestion${row.ai.suggestionCount === 1 ? "" : "s"}`
                        : "Unavailable"}
                      </span>
                      <span>{row.ai.modelList.length > 0 ? row.ai.modelList.join(", ") : "No model recorded"}</span>
                    </div>
                  </div>
                  <div className="logistics-row-runmeta logistics-row-runmeta-corner">
                    <span className="logistics-row-runmeta-label">Run ID</span>
                    <strong>#{row.id}</strong>
                    <span className="logistics-row-runmeta-label">Run Date</span>
                    <time dateTime={row.run.createdAt || ""}>{formatDate(row.run.createdAt)}</time>
                  </div>
                </button>
                <div className="logistics-row-actions">
                  <button
                    type="button"
                    className="icon-delete-button"
                    onClick={() => onDeleteRun?.(row.id)}
                    disabled={deletingRunId === row.id}
                    title={deletingRunId === row.id ? "Deleting log..." : `Delete run #${row.id}`}
                    aria-label={deletingRunId === row.id ? "Deleting run log" : `Delete run #${row.id}`}
                  >
                    {deletingRunId === row.id ? (
                      <span aria-hidden="true">...</span>
                    ) : (
                      <svg viewBox="0 0 24 24" role="presentation" focusable="false" aria-hidden="true">
                        <path d="M9 3.5h6l.7 1.5H20v2H4V5h4.3L9 3.5Zm-2 6h2v8H7v-8Zm4 0h2v8h-2v-8Zm4 0h2v8h-2v-8Z" />
                      </svg>
                    )}
                  </button>
                </div>
              </article>
            ))}
          </div>

          <div className="logistics-footer">
            <span>{rows.length} run log{rows.length === 1 ? "" : "s"} shown</span>
            {hasMore ? (
              <button
                type="button"
                className="secondary-button"
                onClick={onLoadMore}
                disabled={loadingMore || loading}
              >
                {loadingMore ? "Loading..." : "Load More"}
              </button>
            ) : (
              <span className="muted">No more logs.</span>
            )}
          </div>
        </>
      ) : null}
    </section>
  );
}

function LogisticsDetailModal({
  open,
  loading,
  error,
  detail,
  deleting,
  onClose,
  onDelete,
  onShowJson,
}) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function handleEscape(event) {
      if (event.key === "Escape") {
        onClose?.();
      }
    }
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  const run = detail?.run && typeof detail.run === "object" ? detail.run : {};
  const spec = detail?.spec && typeof detail.spec === "object" ? detail.spec : {};
  const auth = detail?.auth && typeof detail.auth === "object" ? detail.auth : {};
  const suite = detail?.suite && typeof detail.suite === "object" ? detail.suite : {};
  const runSummary = normalizeOutcomeSummary(run.summary);
  const testCases = Array.isArray(suite.test_cases) ? suite.test_cases : [];
  const runResults = Array.isArray(detail?.results) ? detail.results : [];
  const llmInsights = Array.isArray(detail?.llm_insights) ? detail.llm_insights : [];
  const resultByTestId = runResults.reduce((acc, row) => {
    const testId = String(row?.test_id || "").trim();
    if (!testId) {
      return acc;
    }
    acc[testId] = row;
    return acc;
  }, {});
  const insightByTestId = llmInsights.reduce((acc, row) => {
    const testId = String(row?.test_id || "").trim();
    if (!testId) {
      return acc;
    }
    if (!acc[testId]) {
      acc[testId] = [];
    }
    acc[testId].push(row);
    return acc;
  }, {});
  const runId = Number(run.id);
  const specId = Number(spec.id);

  return (
    <div
      className="settings-modal-backdrop logistics-modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose?.();
        }
      }}
    >
      <section
        className="panel settings-modal-card logistics-modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="logistics-modal-title"
      >
        <div className="panel-header settings-modal-header logistics-modal-header">
          <div>
            <p className="eyebrow">Run Detail</p>
            <h2 id="logistics-modal-title">{Number.isFinite(runId) ? `Run #${runId}` : "Run Detail"}</h2>
            <p className="muted">
              {spec.title || "Unknown spec"} {Number.isFinite(specId) ? `(spec #${specId})` : ""}
            </p>
          </div>
          <div className="logistics-modal-header-actions">
            <button
              type="button"
              className="danger-button"
              onClick={() => onDelete?.(runId)}
              disabled={!Number.isFinite(runId) || deleting}
            >
              {deleting ? "Deleting..." : "Delete Log"}
            </button>
            <button type="button" className="secondary-button" onClick={onClose}>
              Close
            </button>
          </div>
        </div>

        {error ? <p className="message error">{error}</p> : null}
        {loading ? (
          <div className="empty-state">
            <p>Loading run detail...</p>
            <span>Fetching suite snapshot, results, and AI outputs.</span>
          </div>
        ) : null}

        {!loading ? (
          <div className="logistics-modal-body">
            <div className="stats-grid logistics-modal-stats">
              <StatCard label="Uploaded" value={formatDate(spec.uploaded_at)} accent="accent-green" />
              <StatCard label="Run Time" value={formatDate(run.created_at)} accent="accent-blue" />
              <StatCard label="Duration" value={formatDuration(run.duration_ms)} accent="accent-amber" />
              <StatCard label="Base URL" value={run.base_url || "Unknown"} accent="accent-red" />
            </div>
            <div className="logistics-modal-meta-row">
              <span>
                Auth: {auth.provided ? `Provided (${String(auth.mode || "none").replaceAll("_", " ")})` : "Not provided"}
              </span>
              <span>Summary: {runSummary.passed} passed / {runSummary.failed} failed / {runSummary.skipped} skipped</span>
              <span>AI records: {llmInsights.length}</span>
            </div>

            <div className="logistics-test-list">
              {testCases.length === 0 ? (
                <div className="empty-state">
                  <p>No stored test cases in this run snapshot.</p>
                  <span>Older runs may not have suite snapshots.</span>
                </div>
              ) : (
                testCases.map((testCase, index) => {
                  const testId = String(testCase?.test_id || `case-${index + 1}`);
                  const result = resultByTestId[testId] || null;
                  const insights = Array.isArray(insightByTestId[testId]) ? insightByTestId[testId] : [];
                  const outcome = normalizeRunOutcome(result?.outcome);
                  return (
                    <article key={testId} className="logistics-test-row">
                      <div className="logistics-test-main">
                        <div className="logistics-test-top">
                          <strong>{testCase?.title || testId}</strong>
                          {outcome ? (
                            <span className={`result-pill result-pill-${outcome.toLowerCase()}`}>{outcome}</span>
                          ) : (
                            <span className="run-chip run-chip-skip">No result</span>
                          )}
                        </div>
                        <div className="logistics-test-meta">
                          <span>{String(testCase?.method || "").toUpperCase()} {String(testCase?.path || "")}</span>
                          <span>{testId}</span>
                          <span>{formatDuration(result?.duration_ms)}</span>
                          <span>{insights.length} AI record{insights.length === 1 ? "" : "s"}</span>
                        </div>
                      </div>
                      <div className="logistics-test-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => onShowJson?.(`Test Case JSON: ${testId}`, testCase)}
                        >
                          Case JSON
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => onShowJson?.(`Run Result JSON: ${testId}`, result || {})}
                          disabled={!result}
                        >
                          Result JSON
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => onShowJson?.(`AI Payload JSON: ${testId}`, insights)}
                          disabled={insights.length === 0}
                        >
                          AI JSON
                        </button>
                      </div>
                    </article>
                  );
                })
              )}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function JsonViewerModal({ open, title, value, onClose }) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function handleEscape(event) {
      if (event.key === "Escape") {
        onClose?.();
      }
    }
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="settings-modal-backdrop logistics-json-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose?.();
        }
      }}
    >
      <section
        className="panel settings-modal-card logistics-json-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="logistics-json-title"
      >
        <div className="panel-header settings-modal-header logistics-json-header">
          <h2 id="logistics-json-title">{title || "JSON Viewer"}</h2>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>
        <pre className="logistics-json-pre">{toPrettyJson(value)}</pre>
      </section>
    </div>
  );
}

function ChatPanel({
  selectedEntry,
  specDisplayIdBySpecId,
  messages,
  draft,
  pending,
  onDraftChange,
  onSend,
}) {
  const transcriptEndRef = useRef(null);
  const hasSelectedSpec = Boolean(selectedEntry?.id);
  const selectedSpecId = hasSelectedSpec ? Number(selectedEntry.id) : null;
  const selectedSpecDisplayId = hasSelectedSpec
    ? Number(specDisplayIdBySpecId?.[String(selectedEntry.id)] || 0) || null
    : null;
  const selectedSpecLabel = selectedEntry
    ? `${selectedEntry.title || selectedEntry.filename} (#${selectedSpecDisplayId || selectedEntry.id})`
    : "No spec selected";

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, pending]);

  return (
    <aside className="chat-panel" aria-label="Assistant chat panel">
      <div className="panel-header chat-panel-header">
        <p className="eyebrow">Assistant</p>
        <h2>Chat</h2>
        <p className="muted">
          Assistant answers are strictly grounded to the selected spec, generated tests, and latest run.
        </p>
      </div>

      <div className="chat-context-control">
        <span className="chat-context-label">Context</span>
        <p className="chat-context-static">
          {hasSelectedSpec ? `Selected spec: ${selectedSpecLabel}` : "Select a spec to enable chat"}
        </p>
      </div>

      <div className="chat-thread" role="log" aria-live="polite" aria-label="Chat transcript">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <p>No messages yet.</p>
            <span>
              Ask about endpoint behavior, generated tests, run summaries, or failed test diagnostics for the selected spec.
            </span>
          </div>
        ) : null}

        {messages.map((message) => {
          const contextSpecId = Number(message?.context?.specId);
          const contextDisplayId = Number(specDisplayIdBySpecId?.[String(contextSpecId)] || 0) || null;
          const contextLabel = message?.context?.mode === CHAT_CONTEXT_SPEC && Number.isFinite(contextSpecId)
            ? `Spec #${contextDisplayId || contextSpecId}`
            : (selectedSpecDisplayId !== null
              ? `Spec #${selectedSpecDisplayId}`
              : (selectedSpecId !== null ? `Spec #${selectedSpecId}` : "Spec not selected"));
          const assistantModelName = String(
            message?.meta?.model
            || message?.meta?.modelId
            || "",
          ).trim();
          const roleLabel = message.role === "user"
            ? "You"
            : message.role === "assistant"
              ? (assistantModelName || "Assistant")
              : "System";

          return (
            <article key={message.id} className={`chat-message chat-message-${message.role}`}>
              <div className="chat-message-meta">
                <span>{roleLabel}</span>
                <span>{contextLabel}</span>
                <time dateTime={message.createdAt}>{formatChatTime(message.createdAt)}</time>
              </div>
              <p className="chat-message-body">{message.content}</p>
            </article>
          );
        })}

        {pending ? (
          <div className="chat-typing-indicator" role="status">
            Assistant is thinking
            <span className="chat-typing-dots" aria-hidden="true">...</span>
          </div>
        ) : null}

        <div ref={transcriptEndRef} />
      </div>

      <form className="chat-composer" onSubmit={onSend}>
        <label className="sr-only" htmlFor="chat-composer-input">Message</label>
        <textarea
          id="chat-composer-input"
          className="chat-composer-input"
          value={draft}
          onChange={(event) => onDraftChange?.(event.target.value)}
          rows={3}
          placeholder={
            hasSelectedSpec
              ? "Ask about this spec's endpoints, generated tests, or latest run..."
              : "Select a spec to start a grounded chat..."
          }
          disabled={pending || !hasSelectedSpec}
        />
        <div className="chat-composer-footer">
          <span className="chat-composer-hint">
            {hasSelectedSpec
              ? `Scoped to spec #${selectedSpecDisplayId || selectedSpecId}`
              : "Select a spec to enable chat"}
          </span>
          <button
            type="submit"
            className="primary-button chat-send-button"
            disabled={pending || draft.trim().length === 0 || !hasSelectedSpec}
          >
            {pending ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </aside>
  );
}

function SettingsModal({
  open,
  tab,
  loading,
  busy,
  error,
  settings,
  isAddModelFormOpen,
  addModelForm,
  providerModels,
  providerModelsLoading,
  providerModelsError,
  customInstructionDraft,
  onClose,
  onTabChange,
  onToggleAddModelForm,
  onAddModelFieldChange,
  onAddModelSubmit,
  onSelectModel,
  onDeleteModel,
  onCustomInstructionDraftChange,
  onSaveCustomInstruction,
  onResetCustomInstruction,
}) {
  const [isModelListOpen, setIsModelListOpen] = useState(false);
  const modelListRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function handleEscape(event) {
      if (event.key === "Escape") {
        onClose?.();
      }
    }
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [open, onClose]);

  useEffect(() => {
    if (!open || tab !== SETTINGS_TAB_MODEL) {
      setIsModelListOpen(false);
    }
  }, [open, tab]);

  useEffect(() => {
    if (!isModelListOpen) {
      return undefined;
    }
    function handleOutsideClick(event) {
      if (modelListRef.current && !modelListRef.current.contains(event.target)) {
        setIsModelListOpen(false);
      }
    }
    window.addEventListener("mousedown", handleOutsideClick);
    return () => {
      window.removeEventListener("mousedown", handleOutsideClick);
    };
  }, [isModelListOpen]);

  if (!open) {
    return null;
  }

  const models = Array.isArray(settings?.models) ? settings.models : [];
  const selectedModelId = String(settings?.active_model_id || "");
  const selectedModel = models.find((entry) => String(entry?.id || "") === selectedModelId) || models[0] || null;
  const modelNameCounts = models.reduce((counts, entry) => {
    const key = String(entry?.model || "").trim().toLowerCase();
    if (!key) {
      return counts;
    }
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});

  function getModelOptionLabel(entry) {
    const modelName = String(entry?.model || entry?.label || "").trim() || "Unnamed model";
    const key = modelName.toLowerCase();
    if ((modelNameCounts[key] || 0) <= 1) {
      return modelName;
    }
    const provider = String(entry?.provider || "").trim().toUpperCase();
    return provider ? `${modelName} (${provider})` : modelName;
  }

  const selectedModelLabel = selectedModel ? getModelOptionLabel(selectedModel) : "No models available";
  const addProvider = String(addModelForm?.provider || "").trim().toLowerCase();
  const isExternalProvider = addProvider === "openai" || addProvider === "anthropic";
  const providerModelCatalog = Array.isArray(providerModels) ? providerModels : [];
  const selectedProviderModelExists = providerModelCatalog.some((entry) => entry.id === addModelForm.model);
  const providerModelValue = selectedProviderModelExists ? addModelForm.model : "";

  return (
    <div
      className="settings-modal-backdrop"
      onClick={() => onClose?.()}
      role="presentation"
    >
      <section
        className="panel settings-modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="panel-header settings-modal-header">
          <div>
            <p className="eyebrow">Assistant</p>
            <h2 id="settings-modal-title">LLM Settings</h2>
            <p className="muted">Pick your model target and save optional custom instructions.</p>
          </div>
          <button
            type="button"
            className="secondary-button settings-close-button"
            onClick={() => onClose?.()}
            disabled={busy}
          >
            Close
          </button>
        </div>

        <div className="tab-row" role="tablist" aria-label="Settings sections">
          <button
            type="button"
            className={`tab-button ${tab === SETTINGS_TAB_MODEL ? "active" : ""}`}
            onClick={() => onTabChange?.(SETTINGS_TAB_MODEL)}
          >
            Model
          </button>
          <button
            type="button"
            className={`tab-button ${tab === SETTINGS_TAB_CUSTOM ? "active" : ""}`}
            onClick={() => onTabChange?.(SETTINGS_TAB_CUSTOM)}
          >
            Custom Instructions
          </button>
        </div>

        {error ? <p className="message error settings-inline-message">{error}</p> : null}

        {tab === SETTINGS_TAB_MODEL ? (
          <div className="settings-modal-section">
            {loading ? (
              <p className="muted">Loading settings...</p>
            ) : (
              <>
                <div className="settings-model-picker">
                  <label className="field settings-model-field">
                    <span>Choose your model</span>
                    <div className="settings-model-dropdown" ref={modelListRef}>
                      <button
                        id="settings-model-select"
                        type="button"
                        className="settings-model-select settings-model-trigger"
                        onClick={() => setIsModelListOpen((current) => !current)}
                        aria-haspopup="listbox"
                        aria-expanded={isModelListOpen}
                        disabled={busy || models.length === 0}
                      >
                        <span>{selectedModelLabel}</span>
                        <span className={`settings-model-caret ${isModelListOpen ? "open" : ""}`} aria-hidden="true">
                          v
                        </span>
                      </button>
                      {isModelListOpen ? (
                        <div className="settings-model-list" role="listbox" aria-labelledby="settings-model-select">
                          {models.map((entry) => {
                            const optionLabel = getModelOptionLabel(entry);
                            const isSelected = String(entry.id) === selectedModelId;
                            const canDelete = String(entry?.source || "").trim().toLowerCase() === "user";
                            return (
                              <div
                                key={entry.id}
                                className={`settings-model-list-item ${isSelected ? "active" : ""}`}
                              >
                                <button
                                  type="button"
                                  className="settings-model-option"
                                  role="option"
                                  aria-selected={isSelected}
                                  onClick={() => {
                                    setIsModelListOpen(false);
                                    onSelectModel?.(entry.id);
                                  }}
                                  disabled={busy}
                                >
                                  {optionLabel}
                                </button>
                                {canDelete ? (
                                  <button
                                    type="button"
                                    className="settings-model-delete-button"
                                    onClick={(event) => {
                                      event.preventDefault();
                                      event.stopPropagation();
                                      onDeleteModel?.(entry.id, optionLabel);
                                    }}
                                    aria-label={`Delete ${optionLabel}`}
                                    disabled={busy}
                                  >
                                    <svg viewBox="0 0 24 24" role="presentation" focusable="false" aria-hidden="true">
                                      <path d="M8.5 5.5h7l-.6-1.4a1 1 0 0 0-.9-.6h-4a1 1 0 0 0-.9.6z" />
                                      <path d="M6 7h12l-.7 12a1.5 1.5 0 0 1-1.5 1.4h-7.6a1.5 1.5 0 0 1-1.5-1.4z" />
                                      <path d="M4.5 7h15" />
                                    </svg>
                                  </button>
                                ) : null}
                              </div>
                            );
                          })}
                        </div>
                      ) : null}
                    </div>
                  </label>
                  <button
                    type="button"
                    className="secondary-button settings-add-model-button"
                    onClick={() => onToggleAddModelForm?.()}
                    disabled={busy}
                  >
                    {isAddModelFormOpen ? "Cancel" : "Add model"}
                  </button>
                </div>

                {isAddModelFormOpen ? (
                  <form className="settings-model-form" onSubmit={onAddModelSubmit}>
                    <h3>Add Model</h3>
                    <label className="field">
                      <span>Provider</span>
                      <select
                        value={addModelForm.provider}
                        onChange={(event) => onAddModelFieldChange?.("provider", event.target.value)}
                        disabled={busy}
                      >
                        {SETTINGS_MODEL_PROVIDERS.map((provider) => (
                          <option key={provider.value} value={provider.value}>{provider.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      <span>API key</span>
                      <input
                        type="password"
                        value={addModelForm.api_key}
                        onChange={(event) => onAddModelFieldChange?.("api_key", event.target.value)}
                        placeholder={isExternalProvider ? "Required to load provider models" : "Optional"}
                        disabled={busy}
                      />
                    </label>
                    {isExternalProvider ? (
                      <label className="field">
                        <span>Model</span>
                        <select
                          value={providerModelValue}
                          onChange={(event) => onAddModelFieldChange?.("model", event.target.value)}
                          disabled={busy || providerModelsLoading || providerModelCatalog.length === 0}
                          required
                        >
                          {providerModelsLoading ? <option value="">Loading models...</option> : null}
                          {!providerModelsLoading && providerModelCatalog.length === 0 ? (
                            <option value="">
                              {providerModelsError ? "No models available" : "Enter API key to load models"}
                            </option>
                          ) : null}
                          {!providerModelsLoading
                            ? providerModelCatalog.map((entry) => (
                              <option key={entry.id} value={entry.id}>{entry.label}</option>
                            ))
                            : null}
                        </select>
                      </label>
                    ) : (
                      <label className="field">
                        <span className="settings-model-label-inline">
                          <span>Model</span>
                          <a href="https://ollama.com/library" target="_blank" rel="noreferrer">
                            ollama.com/library
                          </a>
                        </span>
                        <input
                          type="text"
                          value={addModelForm.model}
                          onChange={(event) => onAddModelFieldChange?.("model", event.target.value)}
                          placeholder="e.g. qwen3-coder:latest"
                          disabled={busy}
                          required
                        />
                      </label>
                    )}
                    {providerModelsError ? (
                      <p className="message error settings-inline-message">{providerModelsError}</p>
                    ) : null}
                    <label className="field">
                      <span>Label (optional)</span>
                      <input
                        type="text"
                        value={addModelForm.label}
                        onChange={(event) => onAddModelFieldChange?.("label", event.target.value)}
                        placeholder="Friendly display name"
                        disabled={busy}
                      />
                    </label>
                    <label className="field">
                      <span>Base URL (optional)</span>
                      <input
                        type="text"
                        value={addModelForm.base_url}
                        onChange={(event) => onAddModelFieldChange?.("base_url", event.target.value)}
                        placeholder="Leave blank for provider default"
                        disabled={busy}
                      />
                    </label>
                    <button
                      type="submit"
                      className="primary-button"
                      disabled={busy || (isExternalProvider && providerModelCatalog.length === 0)}
                    >
                      {busy ? "Saving..." : "Add Model"}
                    </button>
                  </form>
                ) : null}
              </>
            )}
          </div>
        ) : (
          <div className="settings-modal-section">
            <div className="settings-custom-block">
              <label className="field" htmlFor="settings-custom-instruction">
                <span>What should the assistant know about your preferences?</span>
                <textarea
                  id="settings-custom-instruction"
                  className="settings-custom-textarea"
                  value={customInstructionDraft}
                  onChange={(event) => onCustomInstructionDraftChange?.(event.target.value)}
                  placeholder="Add your own custom instructions..."
                  rows={7}
                  disabled={busy}
                />
              </label>
              <p className="muted">
                This instruction profile is applied to spec-grounded assistant chat prompts.
              </p>
              <div className="settings-custom-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => onSaveCustomInstruction?.()}
                  disabled={busy}
                >
                  {busy ? "Saving..." : "Save Instructions"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => onResetCustomInstruction?.()}
                  disabled={busy}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function SpecDetails({
  entry,
  displaySpecId,
  runState,
  onUpdateTestCase,
  onResetTestCase,
  onRunTests,
  onExplainFailure,
  onUpdateRunConfig,
  onResetRunConfig,
  onApplySuggestedTest,
}) {
  const generatedCases = Array.isArray(entry?.generatedTests) ? entry.generatedTests : [];
  const uploadDefaultCases = Array.isArray(entry?.uploadDefaultTests)
    ? entry.uploadDefaultTests
    : (Array.isArray(entry?.originalGeneratedTests) ? entry.originalGeneratedTests : generatedCases);
  const originalCases = uploadDefaultCases;
  const runBaselineCases = Array.isArray(runState?.baselineTests) ? runState.baselineTests : [];
  const baselineCases = runBaselineCases.length > 0 ? runBaselineCases : originalCases;
  const [jsonDrafts, setJsonDrafts] = useState({});
  const runConfigs = resolveRunConfigs(entry || null, entry || null);
  const runConfigDefault = runConfigs.runConfigDefault;
  const runConfigCurrent = runConfigs.runConfigCurrent;
  const baseUrlInput = runConfigCurrent.baseUrl;
  const runAuthMode = runConfigCurrent.authMode;
  const runBearerToken = runConfigCurrent.bearerToken;
  const runApiKeyValue = runConfigCurrent.apiKeyValue;
  const runApiKeyHeader = runConfigCurrent.apiKeyHeader;
  const llmByTestId = runState?.llmByTestId && typeof runState.llmByTestId === "object"
    ? runState.llmByTestId
    : {};
  const [llmLoadingDotCount, setLlmLoadingDotCount] = useState(1);
  const isAnyLlmActionRunning = Object.values(llmByTestId).some((caseState) => {
    const action = String(caseState?.action || "");
    return action === "explaining" || action === "suggesting";
  });

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
    if (!isAnyLlmActionRunning) {
      setLlmLoadingDotCount(1);
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setLlmLoadingDotCount((current) => (current >= 3 ? 1 : current + 1));
    }, 420);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isAnyLlmActionRunning]);

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
  const previewTotals = entry?.preview?.totals && typeof entry.preview.totals === "object" ? entry.preview.totals : {};
  const previewCategoryOrder = Object.keys(previewTotals);
  const dynamicCategoryOrder = Object.keys(casesByCategory).filter((key) => !previewCategoryOrder.includes(key));
  const categoryOrder = [...previewCategoryOrder, ...dynamicCategoryOrder];
  const hasJsonDraftErrors = Object.values(jsonDrafts).some(
    (draft) => Boolean(draft?.inputDataError || draft?.expectedResultError),
  );
  const latestRunResults = Array.isArray(runState?.result?.results) ? runState.result.results : [];
  const runSummarySource = runState?.result?.summary || entry?.latestRunSummary || null;
  const hasLatestRun = Boolean(runState?.runId !== null || entry?.latestRunId !== null || runSummarySource || latestRunResults.length > 0);
  const latestRunSummary = hasLatestRun ? resolveRunSummary(runSummarySource, latestRunResults) : null;
  const latestRunId = runState?.runId || entry?.latestRunId || null;
  const hasSpecRunData = Boolean(hasLatestRun && latestRunSummary?.total > 0);
  const methodCounts = entry?.preview?.methodCounts && typeof entry.preview.methodCounts === "object"
    ? entry.preview.methodCounts
    : {};
  const methodEntries = Object.entries(methodCounts);
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
  const isRunConfigAtDefault = deepEqual(runConfigCurrent, runConfigDefault);
  const isPayloadAtDefault = deepEqual(generatedCases, uploadDefaultCases);
  const isRunDefaultsApplied = isRunConfigAtDefault && isPayloadAtDefault;

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
      spec_id: entry?.id ?? null,
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

  function handleExplainFailureCase(testId) {
    if (!entry || !latestRunId || !onExplainFailure) {
      return;
    }
    onExplainFailure(entry.id, latestRunId, testId);
  }

  function handleRunConfigChange(patch) {
    if (!entry || !onUpdateRunConfig || !patch || typeof patch !== "object") {
      return;
    }
    onUpdateRunConfig(entry.id, patch);
  }

  function handleResetRunConfigToDefaults() {
    if (!entry || !onResetRunConfig) {
      return;
    }
    const nextDrafts = {};
    uploadDefaultCases.forEach((testCase, index) => {
      const firstStep = testCase?.steps?.[0] || null;
      nextDrafts[index] = {
        inputDataText: toPrettyJson(firstStep?.input_data ?? null),
        expectedResultText: toPrettyJson(testCase?.expected_result ?? null),
        inputDataError: "",
        expectedResultError: "",
      };
    });
    setJsonDrafts(nextDrafts);
    onResetRunConfig(entry.id);
  }

  function handleApplySuggestedCase(testId) {
    if (!entry || !onApplySuggestedTest) {
      return;
    }
    onApplySuggestedTest(entry.id, testId);
  }

  return (
    <section className="panel details-panel">
      <div className="panel-header">
        <p className="eyebrow">Spec Detail</p>
        <h2>{entry.title || entry.filename}</h2>
        <p className="muted">Review metadata, test coverage overview, and edit generated JSON before running tests.</p>
      </div>

      <div className="spec-overview-grid">
        <OutcomeDonutCard
          title="Latest Run Outcomes"
          summary={latestRunSummary}
          hasData={hasSpecRunData}
          emptyLabel="No run yet"
          totalLabel="Tests in latest run"
        />
        <div className="spec-overview-right">
          <div className="stats-grid spec-meta-grid">
            <StatCard label="Spec ID" value={displaySpecId || "-"} accent="accent-amber" />
            <StatCard label="Version" value={entry.version || "Unknown"} accent="accent-blue" />
            <StatCard label="Uploaded" value={formatDate(entry.created_at)} accent="accent-green" />
            <StatCard label="Endpoints" value={entry.preview?.endpointCount ?? "Unknown"} accent="accent-red" />
          </div>
          <div className="subpanel methods-compact-card">
            <h3>Methods</h3>
            {methodEntries.length > 0 ? (
              <div className="category-list">
                {methodEntries.map(([method, count]) => (
                  <div key={method} className="category-row">
                    <span>{method}</span>
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">Methods will appear after preview data is available.</p>
            )}
          </div>
        </div>
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
                        <span>Total generated tests</span>
                        <strong>{generatedCases.length}</strong>
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
                        <button
                          type="button"
                          className="secondary-button run-reset-button"
                          onClick={handleResetRunConfigToDefaults}
                          disabled={runState?.loading || isRunDefaultsApplied}
                        >
                          Reset Defaults
                        </button>
                        <input
                          type="text"
                          className="base-url-input"
                          placeholder="https://api.example.com"
                          value={baseUrlInput}
                          onChange={(event) => handleRunConfigChange({ baseUrl: event.target.value })}
                          aria-label="Base URL"
                        />
                      </div>
                      <div className="auth-controls">
                        <label className="auth-field">
                          <span className="auth-label">Run Auth</span>
                          <select
                            className="auth-select"
                            value={runAuthMode}
                            onChange={(event) => handleRunConfigChange({ authMode: event.target.value })}
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
                              onChange={(event) => handleRunConfigChange({ bearerToken: event.target.value })}
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
                                onChange={(event) => handleRunConfigChange({ apiKeyValue: event.target.value })}
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
                                onChange={(event) => handleRunConfigChange({ apiKeyHeader: event.target.value })}
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
                      {categoryOrder.map((category) => {
                        const categoryCases = casesByCategory[category] || [];
                        const count = categoryCases.length;
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
                                    const llmState = llmByTestId[testCase?.test_id || ""] || {};
                                    const explanationPayload = llmState?.explanation || null;
                                    const explanationDisplayText = formatExplanationForDisplay(
                                      explanationPayload?.explanation || "",
                                    );
                                    const suggestionPayload = llmState?.suggestion || null;
                                    const llmAction = String(llmState?.action || "");
                                    const llmError = String(llmState?.error || "");
                                    const addedMessage = String(llmState?.addedMessage || "");
                                    const canApplySuggestion = Boolean(
                                      suggestionPayload?.can_apply && suggestionPayload?.suggested_test_case,
                                    );
                                    const isLlmBusy = llmAction === "explaining" || llmAction === "suggesting";
                                    const loadingDots = ".".repeat(llmLoadingDotCount);
                                    const explainButtonLabel = llmAction === "explaining"
                                      ? "Generating Explanation"
                                      : llmAction === "suggesting"
                                        ? "Generating Test Cases"
                                        : "Explain with AI";
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
                                              <div className="llm-actions-row">
                                                <button
                                                  type="button"
                                                  className="secondary-button llm-action-button"
                                                  onClick={() => handleExplainFailureCase(testCase?.test_id || "")}
                                                  disabled={!latestRunId || isLlmBusy}
                                                >
                                                  {isLlmBusy ? (
                                                    <>
                                                      {explainButtonLabel}
                                                      {" "}
                                                      <span className="llm-loading-dots" aria-hidden="true">{loadingDots}</span>
                                                    </>
                                                  ) : explainButtonLabel}
                                                </button>
                                              </div>
                                              {llmError ? <p className="json-error">{llmError}</p> : null}
                                              {explanationDisplayText ? (
                                                <div className="llm-response-block">
                                                  <p className="run-feedback-title">AI Explanation</p>
                                                  <p className="run-feedback-line">{explanationDisplayText}</p>
                                                  {explanationPayload?.warning ? (
                                                    <p className="run-feedback-line llm-warning">{explanationPayload.warning}</p>
                                                  ) : null}
                                                  {explanationPayload?.used_fallback ? (
                                                    <p className="run-feedback-line llm-warning">
                                                      LLM output was unavailable or invalid, so a deterministic fallback explanation was used.
                                                    </p>
                                                  ) : null}
                                                  {explanationPayload?.llm_error ? (
                                                    <details className="run-feedback-details">
                                                      <summary>LLM debug details</summary>
                                                      <pre className="run-feedback-pre">{String(explanationPayload.llm_error)}</pre>
                                                    </details>
                                                  ) : null}
                                                </div>
                                              ) : null}
                                              {suggestionPayload ? (
                                                <div className="llm-response-block">
                                                  <p className="run-feedback-title">Suggested Extra Test</p>
                                                  <p className="run-feedback-line">{suggestionPayload.reason || "No reason returned."}</p>
                                                  {suggestionPayload?.warning ? (
                                                    <p className="run-feedback-line llm-warning">{suggestionPayload.warning}</p>
                                                  ) : null}
                                                  {suggestionPayload?.used_fallback ? (
                                                    <p className="run-feedback-line llm-warning">
                                                      Suggested test is a fallback because model output was unavailable or invalid.
                                                    </p>
                                                  ) : null}
                                                  {suggestionPayload?.llm_error ? (
                                                    <details className="run-feedback-details">
                                                      <summary>LLM debug details</summary>
                                                      <pre className="run-feedback-pre">{String(suggestionPayload.llm_error)}</pre>
                                                    </details>
                                                  ) : null}
                                                  {suggestionPayload?.suggested_test_case ? (
                                                    <details className="run-feedback-details">
                                                      <summary>Suggested test JSON</summary>
                                                      <pre className="run-feedback-pre">
                                                        {toPrettyJson(suggestionPayload.suggested_test_case)}
                                                      </pre>
                                                    </details>
                                                  ) : null}
                                                  {canApplySuggestion ? (
                                                    <div className="llm-apply-row">
                                                      <button
                                                        type="button"
                                                        className="primary-button llm-apply-button"
                                                        onClick={() => handleApplySuggestedCase(testCase?.test_id || "")}
                                                      >
                                                        Add Suggested Test
                                                      </button>
                                                      {addedMessage ? (
                                                        <span className="llm-apply-success">{addedMessage}</span>
                                                      ) : null}
                                                    </div>
                                                  ) : null}
                                                </div>
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
  const [workspaceView, setWorkspaceView] = useState(WORKSPACE_VIEW_DASHBOARD);
  const [isChatCollapsed, setIsChatCollapsed] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking backend...");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyHydrated, setHistoryHydrated] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [clearHistoryLoading, setClearHistoryLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [specHistory, setSpecHistory] = useState([]);
  const [specCache, setSpecCache] = useState(() => loadSpecCache());
  const [selectedSpecId, setSelectedSpecId] = useState(null);
  const [testRunBySpecId, setTestRunBySpecId] = useState({});
  const [chatMessages, setChatMessages] = useState([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatActiveContext, setChatActiveContext] = useState(() => getDefaultChatContext());
  const [chatPending, setChatPending] = useState(false);
  const [chatRuntimeConfig, setChatRuntimeConfig] = useState(() => normalizeChatRuntimeConfig(DEFAULT_CHAT_RUNTIME_CONFIG));
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState(SETTINGS_TAB_MODEL);
  const [llmSettings, setLlmSettings] = useState(() => normalizeLlmSettings(DEFAULT_LLM_SETTINGS));
  const [llmSettingsLoading, setLlmSettingsLoading] = useState(false);
  const [llmSettingsBusy, setLlmSettingsBusy] = useState(false);
  const [llmSettingsError, setLlmSettingsError] = useState("");
  const [customInstructionDraft, setCustomInstructionDraft] = useState("");
  const [isAddModelFormOpen, setIsAddModelFormOpen] = useState(false);
  const [addModelForm, setAddModelForm] = useState({ ...DEFAULT_ADD_MODEL_FORM });
  const [providerModels, setProviderModels] = useState([]);
  const [providerModelsLoading, setProviderModelsLoading] = useState(false);
  const [providerModelsError, setProviderModelsError] = useState("");
  const [logisticsFilterDraft, setLogisticsFilterDraft] = useState(() => ({ ...DEFAULT_LOGISTICS_FILTERS }));
  const [logisticsFilters, setLogisticsFilters] = useState(() => ({ ...DEFAULT_LOGISTICS_FILTERS }));
  const [logisticsRows, setLogisticsRows] = useState([]);
  const [logisticsLoading, setLogisticsLoading] = useState(false);
  const [logisticsLoadingMore, setLogisticsLoadingMore] = useState(false);
  const [logisticsError, setLogisticsError] = useState("");
  const [logisticsMessage, setLogisticsMessage] = useState("");
  const [logisticsHasMore, setLogisticsHasMore] = useState(false);
  const [logisticsCursor, setLogisticsCursor] = useState(null);
  const [logisticsHydrated, setLogisticsHydrated] = useState(false);
  const [logisticsDeletingRunId, setLogisticsDeletingRunId] = useState(null);
  const [logisticsClearing, setLogisticsClearing] = useState(false);
  const [logisticsDetailRunId, setLogisticsDetailRunId] = useState(null);
  const [logisticsDetailLoading, setLogisticsDetailLoading] = useState(false);
  const [logisticsDetailError, setLogisticsDetailError] = useState("");
  const [logisticsDetail, setLogisticsDetail] = useState(null);
  const [jsonViewerState, setJsonViewerState] = useState({
    open: false,
    title: "",
    value: null,
  });
  const providerModelsRequestRef = useRef(0);
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
    if (!session?.userId) {
      setWorkspaceView(WORKSPACE_VIEW_DASHBOARD);
      setIsChatCollapsed(false);
      setChatMessages([]);
      setChatDraft("");
      setChatActiveContext(getDefaultChatContext());
      setChatPending(false);
      setChatRuntimeConfig(normalizeChatRuntimeConfig(DEFAULT_CHAT_RUNTIME_CONFIG));
      setIsSettingsOpen(false);
      setSettingsTab(SETTINGS_TAB_MODEL);
      setLlmSettings(normalizeLlmSettings(DEFAULT_LLM_SETTINGS));
      setLlmSettingsLoading(false);
      setLlmSettingsBusy(false);
      setLlmSettingsError("");
      setCustomInstructionDraft("");
      setIsAddModelFormOpen(false);
      setAddModelForm({ ...DEFAULT_ADD_MODEL_FORM });
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
      setLogisticsFilterDraft({ ...DEFAULT_LOGISTICS_FILTERS });
      setLogisticsFilters({ ...DEFAULT_LOGISTICS_FILTERS });
      setLogisticsRows([]);
      setLogisticsLoading(false);
      setLogisticsLoadingMore(false);
      setLogisticsError("");
      setLogisticsMessage("");
      setLogisticsHasMore(false);
      setLogisticsCursor(null);
      setLogisticsHydrated(false);
      setLogisticsDeletingRunId(null);
      setLogisticsClearing(false);
      setLogisticsDetailRunId(null);
      setLogisticsDetailLoading(false);
      setLogisticsDetailError("");
      setLogisticsDetail(null);
      setJsonViewerState({ open: false, title: "", value: null });
      providerModelsRequestRef.current = 0;
      return;
    }

    const storedThread = loadChatThread(session.userId);
    setChatMessages(storedThread.messages);
    setChatDraft("");
    setChatActiveContext(storedThread.activeContext);
    setChatPending(false);
    setChatRuntimeConfig(storedThread.runtimeConfig);
  }, [session?.userId]);

  useEffect(() => {
    if (!session?.userId) {
      return;
    }

    saveChatThread(session.userId, {
      messages: chatMessages,
      activeContext: chatActiveContext,
      runtimeConfig: chatRuntimeConfig,
    });
  }, [chatActiveContext, chatMessages, chatRuntimeConfig, session?.userId]);

  useEffect(() => {
    if (!session?.token) {
      return undefined;
    }
    let cancelled = false;

    async function hydrateLlmSettings() {
      setLlmSettingsLoading(true);
      setLlmSettingsError("");
      try {
        const payload = await fetchLlmSettings(session.token);
        if (cancelled) {
          return;
        }
        const normalized = normalizeLlmSettings(payload);
        const activeModel = getActiveLlmModelEntry(normalized);
        setLlmSettings(normalized);
        setCustomInstructionDraft(normalized.custom_instruction);
        setChatRuntimeConfig((current) => normalizeChatRuntimeConfig({
          ...current,
          modelId: String(activeModel?.model || current?.modelId || DEFAULT_CHAT_RUNTIME_CONFIG.modelId),
          userInstruction: normalized.custom_instruction,
        }));
      } catch (error) {
        if (!cancelled) {
          setLlmSettingsError(error.message || "Unable to load LLM settings.");
        }
      } finally {
        if (!cancelled) {
          setLlmSettingsLoading(false);
        }
      }
    }

    hydrateLlmSettings();
    return () => {
      cancelled = true;
    };
  }, [session?.token]);

  useEffect(() => {
    if (!isAddModelFormOpen) {
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
      return undefined;
    }

    const provider = String(addModelForm.provider || "").trim().toLowerCase();
    const isSupportedDiscoveryProvider = provider === "openai" || provider === "anthropic";
    if (!isSupportedDiscoveryProvider) {
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
      return undefined;
    }

    const apiKey = String(addModelForm.api_key || "").trim();
    if (!apiKey || !session?.token) {
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
      return undefined;
    }

    const baseUrl = String(addModelForm.base_url || "").trim();
    const requestId = providerModelsRequestRef.current + 1;
    providerModelsRequestRef.current = requestId;
    const timeoutId = window.setTimeout(async () => {
      setProviderModelsLoading(true);
      setProviderModelsError("");
      try {
        const payload = await fetchLlmProviderModels(session.token, provider, {
          api_key: apiKey,
          base_url: baseUrl || undefined,
        });
        if (providerModelsRequestRef.current !== requestId) {
          return;
        }
        const normalizedModels = normalizeProviderModelCatalog(payload?.models);
        setProviderModels(normalizedModels);
        setProviderModelsError(normalizedModels.length > 0 ? "" : "No models were returned for this provider.");
        setAddModelForm((current) => {
          const currentProvider = String(current.provider || "").trim().toLowerCase();
          if (currentProvider !== provider) {
            return current;
          }
          const currentModel = String(current.model || "").trim();
          const currentModelExists = normalizedModels.some((entry) => entry.id === currentModel);
          if (currentModelExists) {
            return current;
          }
          return {
            ...current,
            model: normalizedModels[0]?.id || "",
          };
        });
      } catch (error) {
        if (providerModelsRequestRef.current !== requestId) {
          return;
        }
        setProviderModels([]);
        setProviderModelsError(error.message || "Unable to load provider models.");
      } finally {
        if (providerModelsRequestRef.current === requestId) {
          setProviderModelsLoading(false);
        }
      }
    }, 450);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [
    addModelForm.api_key,
    addModelForm.base_url,
    addModelForm.provider,
    isAddModelFormOpen,
    session?.token,
  ]);

  useEffect(() => {
    if (!session?.token) {
      setSpecHistory([]);
      setSelectedSpecId(null);
      setTestRunBySpecId({});
      setHistoryHydrated(false);
      return;
    }

    setHistoryHydrated(false);
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
              const hasLatestRunField = row && typeof row === "object" && Object.prototype.hasOwnProperty.call(row, "latest_run");
              const latestRunState = buildLatestRunState(
                row?.latest_run,
                hasLatestRunField ? null : (cached || null),
              );
              const generatedSuite = cached?.generatedSuite && typeof cached.generatedSuite === "object"
                ? cached.generatedSuite
                : null;
              const suiteTestCases = Array.isArray(generatedSuite?.test_cases) ? generatedSuite.test_cases : [];
              const generatedTests = Array.isArray(cached?.generatedTests) ? cached.generatedTests : suiteTestCases;
              const originalGeneratedTests = Array.isArray(cached?.originalGeneratedTests)
                ? cached.originalGeneratedTests
                : (Array.isArray(cached?.uploadDefaultTests) ? cached.uploadDefaultTests : generatedTests);
              const uploadDefaultTests = resolveUploadDefaultTests(
                { generatedSuite },
                cached || null,
              );
              const runConfigs = resolveRunConfigs(
                {
                  parsed: cached?.parsed || null,
                  generatedSuite,
                },
                cached || null,
              );
              return {
                ...row,
                created_at: row.created_at || cached?.createdAt || cached?.uploadedAt || null,
                preview: cached?.preview || null,
                parsed: cached?.parsed || null,
                generatedSuite,
                generatedTests,
                originalGeneratedTests,
                uploadDefaultTests,
                runConfigDefault: runConfigs.runConfigDefault,
                runConfigCurrent: runConfigs.runConfigCurrent,
                latestRunId: latestRunState.latestRunId,
                latestRunCreatedAt: latestRunState.latestRunCreatedAt,
                latestRunSummary: latestRunState.latestRunSummary,
                totalCases: generatedTests.length || cached?.preview?.totalCases || 0,
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
          setHistoryHydrated(true);
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
      return getEmptyRunState();
    }
    return testRunBySpecId[selectedEntry.id] || getEmptyRunState();
  }, [selectedEntry, testRunBySpecId]);
  const workspaceRunSummary = useMemo(
    () => aggregateWorkspaceLatestRunSummary(specHistory),
    [specHistory],
  );
  const specDisplayIdBySpecId = useMemo(
    () => specHistory.reduce((acc, entry, index) => {
      acc[String(entry.id)] = index + 1;
      return acc;
    }, {}),
    [specHistory],
  );
  const selectedEntryDisplayId = useMemo(() => {
    if (!selectedEntry) {
      return null;
    }
    return Number(specDisplayIdBySpecId[String(selectedEntry.id)] || 0) || null;
  }, [selectedEntry, specDisplayIdBySpecId]);
  const hasWorkspaceRunData = workspaceRunSummary.total > 0;

  useEffect(() => {
    if (chatActiveContext.mode !== CHAT_CONTEXT_SPEC) {
      return;
    }

    if (!selectedEntry?.id) {
      if (!historyHydrated) {
        return;
      }
      setChatActiveContext(getDefaultChatContext());
      return;
    }

    const selectedId = Number(selectedEntry.id);
    if (!Number.isFinite(selectedId)) {
      setChatActiveContext(getDefaultChatContext());
      return;
    }

    if (selectedId !== Number(chatActiveContext.specId)) {
      setChatActiveContext({
        mode: CHAT_CONTEXT_SPEC,
        specId: selectedId,
      });
    }
  }, [chatActiveContext.mode, chatActiveContext.specId, historyHydrated, selectedEntry?.id]);

  useEffect(() => {
    if (!session?.token || !selectedEntry?.id) {
      return undefined;
    }

    let cancelled = false;

    async function hydrateLatestRun() {
      try {
        const payload = await fetchLatestRunForSpec(session.token, selectedEntry.id);
        if (cancelled) {
          return;
        }

        const artifact = payload?.artifact && typeof payload.artifact === "object" ? payload.artifact : {};
        const parsed = artifact?.parsed && typeof artifact.parsed === "object" ? artifact.parsed : null;
        const generatedSuite = artifact?.generated_suite && typeof artifact.generated_suite === "object"
          ? artifact.generated_suite
          : null;
        const latestRun = payload?.latest_run && typeof payload.latest_run === "object" ? payload.latest_run : null;
        const latestRunState = buildLatestRunState(latestRun);

        setSpecHistory((current) =>
          current.map((entry) => {
            if (entry.id !== selectedEntry.id) {
              return entry;
            }

            const nextParsed = entry.parsed || parsed;
            const nextGeneratedSuite = entry.generatedSuite || generatedSuite;
            const suiteCases = Array.isArray(nextGeneratedSuite?.test_cases) ? nextGeneratedSuite.test_cases : [];
            const hasLocalTests = Array.isArray(entry.generatedTests) && entry.generatedTests.length > 0;
            const nextGeneratedTests = hasLocalTests
              ? entry.generatedTests
              : (cloneJsonValue(suiteCases) || []);
            const nextOriginal = Array.isArray(entry.originalGeneratedTests) && entry.originalGeneratedTests.length > 0
              ? entry.originalGeneratedTests
              : (cloneJsonValue(nextGeneratedTests) || []);
            const nextUploadDefault = Array.isArray(entry.uploadDefaultTests) && entry.uploadDefaultTests.length > 0
              ? entry.uploadDefaultTests
              : resolveUploadDefaultTests(
                  {
                    ...entry,
                    generatedSuite: nextGeneratedSuite,
                  },
                  entry,
                );
            const nextPreview = entry.preview || (nextParsed ? buildSpecPreview(nextParsed) : null);
            const runConfigs = resolveRunConfigs(
              {
                ...entry,
                parsed: nextParsed,
                generatedSuite: nextGeneratedSuite,
              },
              entry,
            );

            return {
              ...entry,
              parsed: nextParsed,
              preview: nextPreview,
              generatedSuite: nextGeneratedSuite,
              generatedTests: nextGeneratedTests,
              originalGeneratedTests: nextOriginal,
              uploadDefaultTests: nextUploadDefault,
              runConfigDefault: runConfigs.runConfigDefault,
              runConfigCurrent: runConfigs.runConfigCurrent,
              latestRunId: latestRunState.latestRunId,
              latestRunCreatedAt: latestRunState.latestRunCreatedAt,
              latestRunSummary: latestRunState.latestRunSummary,
              totalCases: nextPreview?.totalCases || nextGeneratedTests.length || entry.totalCases || 0,
            };
          }),
        );

        if (!latestRun) {
          setTestRunBySpecId((current) => ({
            ...current,
            [selectedEntry.id]: {
              ...(current[selectedEntry.id] || getEmptyRunState()),
              runId: null,
              result: null,
              baselineTests: null,
              llmByTestId: {},
            },
          }));
          return;
        }

        const runSummary = latestRun?.summary && typeof latestRun.summary === "object" ? latestRun.summary : null;
        const runResults = Array.isArray(latestRun?.results) ? latestRun.results : [];
        const runSuite = latestRun?.suite_snapshot && typeof latestRun.suite_snapshot === "object"
          ? latestRun.suite_snapshot
          : null;
        const baselineTests = Array.isArray(runSuite?.test_cases) ? (cloneJsonValue(runSuite.test_cases) || []) : null;

        setTestRunBySpecId((current) => ({
          ...current,
          [selectedEntry.id]: {
            ...getEmptyRunState(),
            result: (runSummary || runResults.length > 0)
              ? { summary: runSummary, results: runResults }
              : null,
            baselineTests,
            runId: latestRun?.id ?? null,
            llmByTestId: {},
          },
        }));
      } catch {
        if (!cancelled) {
          setTestRunBySpecId((current) => ({
            ...current,
            [selectedEntry.id]: getEmptyRunState(),
          }));
        }
      }
    }

    hydrateLatestRun();

    return () => {
      cancelled = true;
    };
  }, [selectedEntry?.id, session?.token]);

  useEffect(() => {
    if (!session?.token || workspaceView !== WORKSPACE_VIEW_LOGS || logisticsHydrated) {
      return;
    }
    void fetchLogisticsRunsPage({
      append: false,
      beforeRunId: null,
      filtersOverride: logisticsFilters,
    });
  }, [session?.token, workspaceView, logisticsHydrated, logisticsFilters.specQuery, logisticsFilters.state]);

  useEffect(() => {
    if (!session?.token || workspaceView !== WORKSPACE_VIEW_LOGS || !logisticsHydrated) {
      return undefined;
    }
    const normalized = normalizeLogisticsFilters(logisticsFilterDraft);
    if (normalized.specQuery === logisticsFilters.specQuery && normalized.state === logisticsFilters.state) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setLogisticsFilters(normalized);
      setLogisticsCursor(null);
      setLogisticsMessage("");
      void fetchLogisticsRunsPage({
        append: false,
        beforeRunId: null,
        filtersOverride: normalized,
      });
    }, normalized.specQuery ? 220 : 0);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [
    logisticsFilterDraft.specQuery,
    logisticsFilterDraft.state,
    logisticsFilters.specQuery,
    logisticsFilters.state,
    logisticsHydrated,
    session?.token,
    workspaceView,
  ]);

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
          totalCases: nextTests.length,
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
        ...buildSpecCacheEntry({
          ...existing,
          generatedTests: nextTests,
        }, existing),
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
        ...buildSpecCacheEntry({
          ...existing,
          generatedTests: updatedTests,
        }, existing),
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
        : (
            Array.isArray(existing.originalGeneratedTests)
              ? existing.originalGeneratedTests
              : (Array.isArray(existing.uploadDefaultTests) ? existing.uploadDefaultTests : currentTests)
          );
      const resetTests = applyGeneratedTestReset(currentTests, originalTests, testIndex);

      userCache[specId] = {
        ...buildSpecCacheEntry({
          ...existing,
          generatedTests: resetTests,
        }, existing),
      };

      return {
        ...current,
        [session.userId]: userCache,
      };
    });
  }

  function handleUpdateRunConfig(specId, patch) {
    if (!patch || typeof patch !== "object" || !specId) {
      return;
    }

    const sourceEntry = specHistory.find((entry) => entry.id === specId) || null;
    const resolvedConfigs = resolveRunConfigs(sourceEntry || null, sourceEntry || null);
    const nextCurrent = normalizeRunConfig(
      { ...resolvedConfigs.runConfigCurrent, ...patch },
      resolvedConfigs.runConfigDefault,
    );

    setSpecHistory((current) =>
      current.map((entry) => (
        entry.id === specId
          ? {
              ...entry,
              runConfigDefault: resolvedConfigs.runConfigDefault,
              runConfigCurrent: nextCurrent,
            }
          : entry
      )),
    );

    if (!session?.userId) {
      return;
    }

    setSpecCache((current) => {
      const userCache = { ...(current[session.userId] || {}) };
      const existing = userCache[specId];
      const seed = existing && typeof existing === "object"
        ? existing
        : {
            filename: sourceEntry?.filename || `spec-${specId}`,
            uploadedAt: sourceEntry?.created_at || null,
            createdAt: sourceEntry?.created_at || null,
          };
      userCache[specId] = {
        ...buildSpecCacheEntry({
          ...seed,
          runConfigDefault: normalizeRunConfig(seed.runConfigDefault, resolvedConfigs.runConfigDefault),
          runConfigCurrent: nextCurrent,
        }, seed),
      };
      return {
        ...current,
        [session.userId]: userCache,
      };
    });
  }

  function handleResetRunConfig(specId) {
    const sourceEntry = specHistory.find((entry) => entry.id === specId) || null;
    const resolvedConfigs = resolveRunConfigs(sourceEntry || null, sourceEntry || null);
    const defaultTests = resolveUploadDefaultTests(sourceEntry || null, sourceEntry || null);

    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }
        const resetTests = cloneJsonValue(defaultTests) || [];
        return {
          ...entry,
          runConfigDefault: resolvedConfigs.runConfigDefault,
          runConfigCurrent: resolvedConfigs.runConfigDefault,
          generatedTests: resetTests,
          totalCases: resetTests.length,
          generatedSuite: entry.generatedSuite
            ? { ...entry.generatedSuite, test_cases: resetTests }
            : entry.generatedSuite,
        };
      }),
    );

    setTestRunBySpecId((current) => ({
      ...current,
      [specId]: {
        ...(current[specId] || getEmptyRunState()),
        llmByTestId: {},
      },
    }));

    if (!session?.userId) {
      return;
    }

    setSpecCache((current) => {
      const userCache = { ...(current[session.userId] || {}) };
      const existing = userCache[specId];
      if (!existing) {
        return current;
      }
      const resetTests = cloneJsonValue(defaultTests) || [];
      userCache[specId] = {
        ...buildSpecCacheEntry({
          ...existing,
          runConfigDefault: normalizeRunConfig(existing.runConfigDefault, resolvedConfigs.runConfigDefault),
          runConfigCurrent: resolvedConfigs.runConfigDefault,
          generatedTests: resetTests,
        }, existing),
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
        ...(current[specId] || getEmptyRunState()),
        loading: true,
        error: "",
      },
    }));

    try {
      const payload = await runGeneratedTests(session.token, suitePayload);
      const executedTests = Array.isArray(suitePayload?.test_cases) ? suitePayload.test_cases : [];
      const latestRunState = buildLatestRunState({
        id: payload?.run_id ?? null,
        created_at: new Date().toISOString(),
        summary: payload?.summary || null,
      });
      setRunBaselineForSpec(specId, executedTests);
      setSpecHistory((current) =>
        current.map((entry) => (
          entry.id === specId
            ? {
                ...entry,
                latestRunId: latestRunState.latestRunId,
                latestRunCreatedAt: latestRunState.latestRunCreatedAt,
                latestRunSummary: latestRunState.latestRunSummary,
              }
            : entry
        )),
      );
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          ...(current[specId] || getEmptyRunState()),
          loading: false,
          error: "",
          result: {
            summary: payload?.summary || null,
            results: Array.isArray(payload?.results) ? payload.results : [],
          },
          baselineTests: cloneJsonValue(executedTests) || [],
          runId: payload?.run_id ?? null,
          llmByTestId: {},
        },
      }));
      setLogisticsHydrated(false);
      if (workspaceView === WORKSPACE_VIEW_LOGS) {
        await fetchLogisticsRunsPage({
          append: false,
          beforeRunId: null,
          filtersOverride: logisticsFilters,
        });
      }
    } catch (error) {
      setTestRunBySpecId((current) => ({
        ...current,
        [specId]: {
          ...(current[specId] || getEmptyRunState()),
          loading: false,
          error: error.message,
        },
      }));
    }
  }

  function setLlmCaseState(specId, testId, updater) {
    setTestRunBySpecId((current) => {
      const existingRun = current[specId] || getEmptyRunState();
      const currentCaseState = existingRun.llmByTestId?.[testId] || {};
      const nextCaseState = typeof updater === "function"
        ? updater(currentCaseState)
        : { ...currentCaseState, ...(updater || {}) };
      return {
        ...current,
        [specId]: {
          ...existingRun,
          llmByTestId: {
            ...(existingRun.llmByTestId || {}),
            [testId]: nextCaseState,
          },
        },
      };
    });
  }

  function syncLogisticsAiForRun(runId, analysisPayload) {
    const normalizedRunId = Number(runId);
    if (!Number.isFinite(normalizedRunId)) {
      return;
    }
    const explanationPayload = analysisPayload?.explanation && typeof analysisPayload.explanation === "object"
      ? analysisPayload.explanation
      : {};
    const suggestionPayload = analysisPayload?.suggestion && typeof analysisPayload.suggestion === "object"
      ? analysisPayload.suggestion
      : {};
    const nextModels = [
      explanationPayload?.model,
      suggestionPayload?.model,
    ]
      .map((value) => String(value || "").trim())
      .filter(Boolean);

    setLogisticsRows((current) =>
      current.map((row) => {
        if (Number(row?.id) !== normalizedRunId) {
          return row;
        }
        const currentModels = Array.isArray(row?.ai?.modelList) ? row.ai.modelList : [];
        const mergedModels = Array.from(new Set([...currentModels, ...nextModels])).filter(Boolean);
        return {
          ...row,
          ai: {
            ...(row.ai || {}),
            hasAny: true,
            hasExplanations: true,
            hasSuggestions: true,
            explanationCount: Math.max(1, toSafeCount(row?.ai?.explanationCount)),
            suggestionCount: Math.max(1, toSafeCount(row?.ai?.suggestionCount)),
            modelList: mergedModels,
          },
        };
      }),
    );
  }

  async function handleExplainFailure(specId, runId, testId) {
    if (!session?.token || !runId || !testId) {
      return;
    }

    setLlmCaseState(specId, testId, (currentCase) => ({
      ...currentCase,
      action: "explaining",
      error: "",
      addedMessage: "",
      addedSuggestionFingerprint: "",
    }));

    try {
      const analysisResponse = await requestLlmFailureAnalysis(session.token, runId, testId);
      const analysisPayload = analysisResponse?.payload && typeof analysisResponse.payload === "object"
        ? analysisResponse.payload
        : null;
      const explanationPayload = analysisPayload?.explanation && typeof analysisPayload.explanation === "object"
        ? analysisPayload.explanation
        : null;
      const suggestionPayload = analysisPayload?.suggestion && typeof analysisPayload.suggestion === "object"
        ? analysisPayload.suggestion
        : null;
      if (!explanationPayload) {
        throw new Error("Explanation payload missing from backend response.");
      }
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        action: "",
        error: "",
        explanation: explanationPayload,
        suggestion: suggestionPayload,
        addedMessage: "",
        addedSuggestionFingerprint: "",
      }));
      syncLogisticsAiForRun(runId, analysisPayload);
      if (logisticsHydrated) {
        void fetchLogisticsRunsPage({
          append: false,
          beforeRunId: null,
          filtersOverride: logisticsFilters,
        });
      }
    } catch (error) {
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        action: "",
        error: error.message || "Unable to generate AI explanation.",
        addedMessage: "",
        addedSuggestionFingerprint: "",
      }));
    }
  }

  function handleApplySuggestedTest(specId, testId) {
    const runState = testRunBySpecId[specId] || getEmptyRunState();
    const suggestionPayload = runState?.llmByTestId?.[testId]?.suggestion;
    const suggestedCase = suggestionPayload?.suggested_test_case;
    if (!suggestedCase) {
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        error: "No suggested test is available to add.",
      }));
      return;
    }

    const sourceEntry = specHistory.find((entry) => entry.id === specId) || null;
    const sourceTests = Array.isArray(sourceEntry?.generatedTests) ? sourceEntry.generatedTests : [];
    const sourceCase = sourceTests.find((testCase) => String(testCase?.test_id || "") === String(testId)) || null;
    const originCategory = sourceCase?.category ? String(sourceCase.category) : "";
    const normalizedCase = normalizeSuggestedTestCaseForSuite(suggestedCase, sourceTests, originCategory);
    if (!normalizedCase) {
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        error: "Suggested test format is invalid.",
      }));
      return;
    }

    if (hasSuggestedCaseAlreadyBeenAdded(sourceTests, normalizedCase)) {
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        error: "This suggested test is already in the payload.",
      }));
      return;
    }

    const normalizedSignature = buildSuggestedCaseSignature(normalizedCase);
    const existingCaseState = runState?.llmByTestId?.[testId] || {};
    if (
      normalizedSignature
      && String(existingCaseState?.addedSuggestionFingerprint || "") === normalizedSignature
    ) {
      setLlmCaseState(specId, testId, (currentCase) => ({
        ...currentCase,
        error: "This suggested test is already in the payload.",
      }));
      return;
    }
    const addedMessage = `${normalizedCase.title} has been added to payload`;
    const normalizedCaseCopy = cloneJsonValue(normalizedCase) || normalizedCase;

    setSpecHistory((current) =>
      current.map((entry) => {
        if (entry.id !== specId) {
          return entry;
        }
        const currentTests = Array.isArray(entry.generatedTests) ? entry.generatedTests : [];
        const nextTests = [...currentTests, normalizedCaseCopy];
        return {
          ...entry,
          generatedTests: nextTests,
          totalCases: nextTests.length,
          generatedSuite: entry.generatedSuite
            ? { ...entry.generatedSuite, test_cases: nextTests }
            : entry.generatedSuite,
        };
      }),
    );

    if (session?.userId) {
      setSpecCache((current) => {
        const userCache = { ...(current[session.userId] || {}) };
        const existing = userCache[specId];
        if (!existing) {
          return current;
        }
        const currentTests = Array.isArray(existing.generatedTests) ? existing.generatedTests : [];
        const nextTests = [...currentTests, normalizedCaseCopy];
        userCache[specId] = {
          ...buildSpecCacheEntry({
            ...existing,
            generatedTests: nextTests,
          }, existing),
        };
        return {
          ...current,
          [session.userId]: userCache,
        };
      });
    }

    setLlmCaseState(specId, testId, (currentCase) => ({
      ...currentCase,
      error: "",
      addedMessage,
      addedSuggestionFingerprint: normalizedSignature,
    }));
  }

  async function refreshSpecLatestRunFromServer(specId) {
    if (!session?.token) {
      return;
    }
    const normalizedSpecId = Number(specId);
    if (!Number.isFinite(normalizedSpecId)) {
      return;
    }
    try {
      const payload = await fetchLatestRunForSpec(session.token, normalizedSpecId);
      const latestRun = payload?.latest_run && typeof payload.latest_run === "object" ? payload.latest_run : null;
      const latestRunState = buildLatestRunState(latestRun);
      setSpecHistory((current) =>
        current.map((entry) => (
          Number(entry.id) === normalizedSpecId
            ? {
                ...entry,
                latestRunId: latestRunState.latestRunId,
                latestRunCreatedAt: latestRunState.latestRunCreatedAt,
                latestRunSummary: latestRunState.latestRunSummary,
              }
            : entry
        )),
      );
      if (Number(selectedSpecId) === normalizedSpecId) {
        if (!latestRun) {
          setTestRunBySpecId((current) => ({
            ...current,
            [normalizedSpecId]: {
              ...(current[normalizedSpecId] || getEmptyRunState()),
              runId: null,
              result: null,
              baselineTests: null,
              llmByTestId: {},
            },
          }));
          return;
        }
        const runSummary = latestRun?.summary && typeof latestRun.summary === "object" ? latestRun.summary : null;
        const runResults = Array.isArray(latestRun?.results) ? latestRun.results : [];
        const runSuite = latestRun?.suite_snapshot && typeof latestRun.suite_snapshot === "object"
          ? latestRun.suite_snapshot
          : null;
        const baselineTests = Array.isArray(runSuite?.test_cases) ? (cloneJsonValue(runSuite.test_cases) || []) : null;
        setTestRunBySpecId((current) => ({
          ...current,
          [normalizedSpecId]: {
            ...getEmptyRunState(),
            result: (runSummary || runResults.length > 0)
              ? { summary: runSummary, results: runResults }
              : null,
            baselineTests,
            runId: latestRun?.id ?? null,
            llmByTestId: {},
          },
        }));
      }
    } catch {
      // Keep the current local snapshot if latest-run hydration fails.
    }
  }

  async function fetchLogisticsRunsPage({ append = false, beforeRunId = null, filtersOverride = null } = {}) {
    if (!session?.token) {
      return;
    }
    const activeFilters = normalizeLogisticsFilters(filtersOverride || logisticsFilters);
    if (append) {
      setLogisticsLoadingMore(true);
    } else {
      setLogisticsLoading(true);
      setLogisticsError("");
    }

    try {
      const payload = await listLogisticsRuns(session.token, {
        limit: LOGISTICS_PAGE_LIMIT,
        before_run_id: beforeRunId || undefined,
        spec_query: activeFilters.specQuery || undefined,
        state: activeFilters.state,
      });
      const normalizedRows = normalizeLogisticsRunCards(payload?.items);
      const nextCursor = Number(payload?.next_before_run_id);
      setLogisticsRows((current) => {
        const merged = append ? [...current, ...normalizedRows] : normalizedRows;
        const byId = new Map();
        for (const row of merged) {
          byId.set(row.id, row);
        }
        return Array.from(byId.values()).sort((left, right) => Number(right.id) - Number(left.id));
      });
      setLogisticsHasMore(Boolean(payload?.has_more));
      setLogisticsCursor(Number.isFinite(nextCursor) ? nextCursor : null);
      setLogisticsHydrated(true);
    } catch (error) {
      if (!append) {
        setLogisticsRows([]);
      }
      setLogisticsError(error.message || "Unable to load run logs.");
      setLogisticsHasMore(false);
      setLogisticsCursor(null);
      setLogisticsHydrated(true);
    } finally {
      if (append) {
        setLogisticsLoadingMore(false);
      } else {
        setLogisticsLoading(false);
      }
    }
  }

  function handleWorkspaceViewChange(nextView) {
    const target = nextView === WORKSPACE_VIEW_LOGS ? WORKSPACE_VIEW_LOGS : WORKSPACE_VIEW_DASHBOARD;
    if (target === workspaceView) {
      return;
    }
    setWorkspaceView(target);
  }

  function handleLogisticsFilterDraftChange(patch) {
    if (!patch || typeof patch !== "object") {
      return;
    }
    setLogisticsFilterDraft((current) => normalizeLogisticsFilters({ ...current, ...patch }));
  }

  async function handleLoadMoreLogistics() {
    if (!logisticsHasMore || logisticsLoading || logisticsLoadingMore || !logisticsCursor) {
      return;
    }
    await fetchLogisticsRunsPage({
      append: true,
      beforeRunId: logisticsCursor,
      filtersOverride: logisticsFilters,
    });
  }

  async function handleOpenLogisticsDetail(runId) {
    if (!session?.token) {
      return;
    }
    const normalizedRunId = Number(runId);
    if (!Number.isFinite(normalizedRunId)) {
      return;
    }
    setLogisticsDetailRunId(normalizedRunId);
    setLogisticsDetailLoading(true);
    setLogisticsDetailError("");
    setLogisticsDetail(null);
    try {
      const payload = await fetchLogisticsRunDetail(session.token, normalizedRunId);
      setLogisticsDetail(payload && typeof payload === "object" ? payload : null);
    } catch (error) {
      setLogisticsDetailError(error.message || "Unable to load run detail.");
    } finally {
      setLogisticsDetailLoading(false);
    }
  }

  function handleCloseLogisticsDetail() {
    setLogisticsDetailRunId(null);
    setLogisticsDetailLoading(false);
    setLogisticsDetailError("");
    setLogisticsDetail(null);
  }

  function handleOpenJsonViewer(title, value) {
    setJsonViewerState({
      open: true,
      title: String(title || "JSON Viewer"),
      value: cloneJsonValue(value),
    });
  }

  function handleCloseJsonViewer() {
    setJsonViewerState({
      open: false,
      title: "",
      value: null,
    });
  }

  async function handleDeleteLogisticsRun(targetRunId) {
    if (!session?.token) {
      return;
    }
    const normalizedRunId = Number(targetRunId);
    if (!Number.isFinite(normalizedRunId)) {
      return;
    }
    const shouldDelete = window.confirm(
      `Delete run #${normalizedRunId}? This will remove the saved run log and AI outputs for this run.`,
    );
    if (!shouldDelete) {
      return;
    }

    const rowToDelete = logisticsRows.find((row) => Number(row.id) === normalizedRunId) || null;
    setLogisticsDeletingRunId(normalizedRunId);
    setLogisticsError("");
    setLogisticsMessage("");
    try {
      await deleteLogisticsRun(session.token, normalizedRunId);
      setLogisticsMessage(`Deleted run #${normalizedRunId}.`);
      setLogisticsRows((current) => current.filter((row) => Number(row.id) !== normalizedRunId));
      if (Number(logisticsDetailRunId) === normalizedRunId) {
        handleCloseLogisticsDetail();
      }
      if (Number.isFinite(Number(rowToDelete?.spec?.id))) {
        await refreshSpecLatestRunFromServer(Number(rowToDelete.spec.id));
      }
      await fetchLogisticsRunsPage({
        append: false,
        beforeRunId: null,
        filtersOverride: logisticsFilters,
      });
    } catch (error) {
      setLogisticsError(error.message || "Unable to delete run log.");
    } finally {
      setLogisticsDeletingRunId(null);
    }
  }

  async function handleClearLogisticsRuns() {
    if (!session?.token || logisticsClearing) {
      return;
    }
    const shouldClear = window.confirm(
      "Clear all run logs and saved AI insights for your account? Uploaded specs will remain.",
    );
    if (!shouldClear) {
      return;
    }

    setLogisticsClearing(true);
    setLogisticsError("");
    setLogisticsMessage("");
    try {
      const payload = await clearLogisticsRuns(session.token);
      const deletedRuns = toSafeCount(payload?.deleted_runs);
      const deletedInsights = toSafeCount(payload?.deleted_llm_insights);
      setLogisticsRows([]);
      setLogisticsHasMore(false);
      setLogisticsCursor(null);
      setLogisticsHydrated(true);
      setLogisticsMessage(
        `Cleared ${deletedRuns} run log${deletedRuns === 1 ? "" : "s"} and ${deletedInsights} AI insight${deletedInsights === 1 ? "" : "s"}.`,
      );
      handleCloseLogisticsDetail();
      handleCloseJsonViewer();
      setSpecHistory((current) =>
        current.map((entry) => ({
          ...entry,
          latestRunId: null,
          latestRunCreatedAt: null,
          latestRunSummary: null,
        })),
      );
      setTestRunBySpecId({});
    } catch (error) {
      setLogisticsError(error.message || "Unable to clear run logs.");
    } finally {
      setLogisticsClearing(false);
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
    setWorkspaceView(WORKSPACE_VIEW_DASHBOARD);
    setIsChatCollapsed(false);
    setSpecHistory([]);
    setSelectedSpecId(null);
    setTestRunBySpecId({});
    setClearHistoryLoading(false);
    setUploadError("");
    setUploadMessage("");
    setLogisticsFilterDraft({ ...DEFAULT_LOGISTICS_FILTERS });
    setLogisticsFilters({ ...DEFAULT_LOGISTICS_FILTERS });
    setLogisticsRows([]);
    setLogisticsLoading(false);
    setLogisticsLoadingMore(false);
    setLogisticsError("");
    setLogisticsMessage("");
    setLogisticsHasMore(false);
    setLogisticsCursor(null);
    setLogisticsHydrated(false);
    setLogisticsDeletingRunId(null);
    setLogisticsClearing(false);
    setLogisticsDetailRunId(null);
    setLogisticsDetailLoading(false);
    setLogisticsDetailError("");
    setLogisticsDetail(null);
    setJsonViewerState({ open: false, title: "", value: null });
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
      setChatMessages([]);
      setChatDraft("");
      setChatActiveContext(getDefaultChatContext());
      setChatPending(false);
      setUploadMessage("");
      setUploadError("");
      setLogisticsRows([]);
      setLogisticsHasMore(false);
      setLogisticsCursor(null);
      setLogisticsHydrated(false);
      setLogisticsError("");
      setLogisticsMessage("");
      setLogisticsDetailRunId(null);
      setLogisticsDetail(null);
      setLogisticsDetailError("");
      handleCloseJsonViewer();
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
          const uploadDefaultTests = cloneJsonValue(rawGeneratedTests) || [];
          const generatedTests = cloneJsonValue(rawGeneratedTests) || [];
          const originalGeneratedTests = cloneJsonValue(rawGeneratedTests) || [];
          const runConfigDefault = buildDefaultRunConfig({ parsed, generatedSuite });
          const runConfigCurrent = normalizeRunConfig(runConfigDefault, runConfigDefault);
          const createdAt = new Date().toISOString();
          lastUploadedId = payload?.id ?? lastUploadedId;

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
            uploadDefaultTests,
            runConfigDefault,
            runConfigCurrent,
            latestRunId: null,
            latestRunCreatedAt: null,
            latestRunSummary: null,
            totalCases: generatedTests.length || preview?.totalCases || 0,
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

      setSpecCache((current) => {
        const next = { ...current };
        const nextUserCache = { ...(next[session.userId] || {}) };
        for (const entry of uploadedEntries) {
          nextUserCache[entry.id] = buildSpecCacheEntry({
            filename: entry.filename,
            uploadedAt: entry.created_at,
            createdAt: entry.created_at,
            preview: entry.preview,
            generatedTests: entry.generatedTests,
            uploadDefaultTests: entry.uploadDefaultTests,
            runConfigDefault: entry.runConfigDefault,
            runConfigCurrent: entry.runConfigCurrent,
          });
        }
        next[session.userId] = nextUserCache;
        return next;
      });
      setSpecHistory((current) => {
        const byId = new Map(current.map((entry) => [entry.id, entry]));
        for (const entry of uploadedEntries) {
          byId.set(entry.id, entry);
        }

        return Array.from(byId.values()).sort((a, b) => Number(b.id) - Number(a.id));
      });
      setTestRunBySpecId((current) => {
        const next = { ...current };
        for (const entry of uploadedEntries) {
          next[entry.id] = getEmptyRunState();
        }
        return next;
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

  function applyLlmSettingsSnapshot(payload) {
    const normalized = normalizeLlmSettings(payload);
    const activeModel = getActiveLlmModelEntry(normalized);
    setLlmSettings(normalized);
    setCustomInstructionDraft(normalized.custom_instruction);
    setChatRuntimeConfig((current) => normalizeChatRuntimeConfig({
      ...current,
      modelId: String(activeModel?.model || current?.modelId || DEFAULT_CHAT_RUNTIME_CONFIG.modelId),
      userInstruction: normalized.custom_instruction,
    }));
  }

  async function mutateLlmSettings(action) {
    if (!session?.token) {
      return false;
    }
    setLlmSettingsBusy(true);
    setLlmSettingsError("");
    try {
      const payload = await action(session.token);
      applyLlmSettingsSnapshot(payload);
      return true;
    } catch (error) {
      setLlmSettingsError(error.message || "Unable to update settings.");
      return false;
    } finally {
      setLlmSettingsBusy(false);
    }
  }

  function handleAddModelFieldChange(field, value) {
    setAddModelForm((current) => {
      const next = {
        ...current,
        [field]: value,
      };
      if (field === "provider") {
        next.model = "";
      }
      return next;
    });
    if (field === "provider") {
      providerModelsRequestRef.current += 1;
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
    } else if (field === "api_key" || field === "base_url") {
      setProviderModelsError("");
    }
  }

  async function handleAddModelSubmit(event) {
    event.preventDefault();
    if (!session?.token) {
      return;
    }

    const provider = String(addModelForm.provider || "").trim().toLowerCase();
    const model = String(addModelForm.model || "").trim();
    const label = String(addModelForm.label || "").trim();
    const baseUrl = String(addModelForm.base_url || "").trim();
    const apiKey = String(addModelForm.api_key || "").trim();
    if (!provider || !model) {
      setLlmSettingsError("Provider and model are required.");
      return;
    }
    if ((provider === "openai" || provider === "anthropic") && !apiKey) {
      setLlmSettingsError("An API key is required for OpenAI and Anthropic models.");
      return;
    }

    const added = await mutateLlmSettings((token) => addLlmModel(token, {
      provider,
      model,
      label,
      base_url: baseUrl,
      api_key: apiKey,
    }));
    if (added) {
      setAddModelForm({ ...DEFAULT_ADD_MODEL_FORM });
      setIsAddModelFormOpen(false);
      setProviderModels([]);
      setProviderModelsLoading(false);
      setProviderModelsError("");
      providerModelsRequestRef.current += 1;
    }
  }

  async function handleDeleteModel(modelId, modelLabel = "") {
    const targetModelId = String(modelId || "").trim();
    if (!targetModelId || !session?.token) {
      return;
    }
    const label = String(modelLabel || "").trim() || "this model";
    const shouldDelete = window.confirm(`Delete ${label}?`);
    if (!shouldDelete) {
      return;
    }
    await mutateLlmSettings((token) => deleteLlmModel(token, targetModelId));
  }

  async function handleSelectModel(modelId) {
    const nextModelId = String(modelId || "").trim();
    if (!nextModelId || !session?.token) {
      return;
    }
    await mutateLlmSettings((token) => updateLlmSettings(token, { active_model_id: nextModelId }));
  }

  async function handleSaveCustomInstruction() {
    if (!session?.token) {
      return;
    }
    await mutateLlmSettings((token) => updateLlmSettings(token, { custom_instruction: customInstructionDraft }));
  }

  function handleResetCustomInstruction() {
    setCustomInstructionDraft(String(llmSettings?.custom_instruction || ""));
  }

  async function handleChatSend(event) {
    event.preventDefault();

    if (!session?.userId || !session?.token || chatPending) {
      return;
    }

    if (!selectedEntry?.id) {
      const noSpecMessage = createChatMessage({
        role: "system",
        content: "Select a specification before sending chat messages.",
        context: getDefaultChatContext(),
        meta: {
          level: "error",
        },
      });
      setChatMessages((current) => [...current, noSpecMessage]);
      return;
    }

    const trimmedDraft = chatDraft.trim();
    if (!trimmedDraft) {
      return;
    }

    const resolvedContext = {
      mode: CHAT_CONTEXT_SPEC,
      specId: Number(selectedEntry.id),
    };
    setChatActiveContext(resolvedContext);

    const priorThread = chatMessages
      .filter((message) => (
        (message?.role === "user" || message?.role === "assistant")
        && Number(message?.context?.specId) === Number(selectedEntry.id)
      ))
      .map((message) => ({
        role: message.role,
        content: String(message.content || "").trim(),
      }))
      .filter((message) => message.content.length > 0)
      .slice(-12);

    const hasLatestRunContext = (
      selectedRunState?.runId !== null
      || Boolean(selectedRunState?.result?.summary)
      || (Array.isArray(selectedRunState?.result?.results) && selectedRunState.result.results.length > 0)
      || (Array.isArray(selectedRunState?.baselineTests) && selectedRunState.baselineTests.length > 0)
    );
    const latestRunSnapshot = hasLatestRunContext
      ? {
          run_id: selectedRunState?.runId ?? null,
          summary: selectedRunState?.result?.summary || {},
          results: Array.isArray(selectedRunState?.result?.results)
            ? (cloneJsonValue(selectedRunState.result.results) || [])
            : [],
          baseline_tests: Array.isArray(selectedRunState?.baselineTests)
            ? (cloneJsonValue(selectedRunState.baselineTests) || [])
            : [],
        }
      : null;

    const userMessage = createChatMessage({
      role: "user",
      content: trimmedDraft,
      context: resolvedContext,
      meta: {
        source: "composer",
      },
    });
    setChatMessages((current) => [...current, userMessage]);
    setChatDraft("");
    setChatPending(true);

    try {
      const response = await requestSpecGroundedChat(session.token, {
        spec_id: Number(selectedEntry.id),
        message: trimmedDraft,
        thread: priorThread,
        context_snapshot: {
          parsed_spec: selectedEntry?.parsed && typeof selectedEntry.parsed === "object"
            ? (cloneJsonValue(selectedEntry.parsed) || selectedEntry.parsed)
            : null,
          generated_tests: Array.isArray(selectedEntry?.generatedTests)
            ? (cloneJsonValue(selectedEntry.generatedTests) || [])
            : [],
          latest_run: latestRunSnapshot,
        },
      });
      const assistantMessage = normalizeChatMessage(response?.assistantMessage)
        || createChatMessage({
          role: "assistant",
          content: "Assistant returned an invalid response payload.",
          context: resolvedContext,
          meta: response?.meta || {},
        });
      setChatMessages((current) => [...current, assistantMessage]);
    } catch (error) {
      const errorMessage = createChatMessage({
        role: "system",
        content: error?.message || "Unable to send chat message.",
        context: resolvedContext,
        meta: {
          level: "error",
        },
      });
      setChatMessages((current) => [...current, errorMessage]);
    } finally {
      setChatPending(false);
    }
  }

  function handleThemeToggle() {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }

  function handleToggleChatPanel() {
    setIsChatCollapsed((current) => !current);
  }

  const isLogsView = workspaceView === WORKSPACE_VIEW_LOGS;
  const showChatPanel = Boolean(session) && !isChatCollapsed;
  const appShellClassName = `app-shell ${showChatPanel ? "" : "chat-rail-collapsed"}`.trim();

  return (
    <div className={appShellClassName} data-theme={theme}>
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
          <div className="hero-controls">
            {session ? (
              <>
                <div className="workspace-segmented-toggle" role="tablist" aria-label="Workspace views">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={workspaceView === WORKSPACE_VIEW_DASHBOARD}
                    className={`workspace-segment-button ${workspaceView === WORKSPACE_VIEW_DASHBOARD ? "active" : ""}`}
                    onClick={() => handleWorkspaceViewChange(WORKSPACE_VIEW_DASHBOARD)}
                  >
                    Dashboard
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={workspaceView === WORKSPACE_VIEW_LOGS}
                    className={`workspace-segment-button ${workspaceView === WORKSPACE_VIEW_LOGS ? "active" : ""}`}
                    onClick={() => handleWorkspaceViewChange(WORKSPACE_VIEW_LOGS)}
                  >
                    Logs
                  </button>
                </div>
                <button
                  type="button"
                  className="settings-trigger-button"
                  onClick={() => {
                    setSettingsTab(SETTINGS_TAB_MODEL);
                    setLlmSettingsError("");
                    setIsAddModelFormOpen(false);
                    setAddModelForm({ ...DEFAULT_ADD_MODEL_FORM });
                    setProviderModels([]);
                    setProviderModelsLoading(false);
                    setProviderModelsError("");
                    providerModelsRequestRef.current += 1;
                    setIsSettingsOpen(true);
                  }}
                  aria-label="Open settings"
                >
                  Settings
                </button>
              </>
            ) : null}
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
          {session ? (
            <button
              type="button"
              className={`chat-bubble-toggle ${showChatPanel ? "active" : ""}`.trim()}
              onClick={handleToggleChatPanel}
              aria-label={showChatPanel ? "Collapse chat panel" : "Open chat panel"}
              title={showChatPanel ? "Collapse chat" : "Open chat"}
            >
              <svg viewBox="0 0 24 24" role="presentation" focusable="false" aria-hidden="true">
                <path d="M4 4.8h16v10.9H8.8L5.2 19V15.7H4V4.8Zm2 2v7h0.8l0.4 0.4L9.7 13.7H18V6.8H6Z" />
              </svg>
            </button>
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
        <main className={`dashboard-grid ${isLogsView ? "dashboard-grid-logs" : ""}`.trim()}>
          {isLogsView ? (
            <LogisticsPanel
              rows={logisticsRows}
              loading={logisticsLoading}
              loadingMore={logisticsLoadingMore}
              error={logisticsError}
              message={logisticsMessage}
              filterDraft={logisticsFilterDraft}
              onFilterDraftChange={handleLogisticsFilterDraftChange}
              hasMore={logisticsHasMore}
              onLoadMore={handleLoadMoreLogistics}
              onOpenDetail={handleOpenLogisticsDetail}
              onDeleteRun={handleDeleteLogisticsRun}
              deletingRunId={logisticsDeletingRunId}
              onClearAll={handleClearLogisticsRuns}
              clearLoading={logisticsClearing}
            />
          ) : (
            <>
              <section className="panel summary-panel">
                <div className="panel-header">
                  <p className="eyebrow">Workspace</p>
                  <h2>Dashboard</h2>
                  <p className="muted">Manage your uploads and inspect each API specification.</p>
                </div>

                <div className="workspace-overview-grid">
                  <OutcomeDonutCard
                    title="Workspace Run Outcomes"
                    summary={workspaceRunSummary}
                    hasData={hasWorkspaceRunData}
                    emptyLabel="No runs yet"
                    totalLabel="Total Runs"
                  />
                  <div className="stats-grid workspace-stats-grid">
                    <StatCard label="Total Uploads" value={historyLoading ? "..." : specHistory.length} accent="accent-blue" />
                    <StatCard
                      label="Specs With Runs"
                      value={historyLoading ? "..." : workspaceRunSummary.specsWithRuns}
                      accent="accent-green"
                    />
                    <StatCard
                      label="Total Runs"
                      value={historyLoading ? "..." : workspaceRunSummary.total}
                      accent="accent-red"
                    />
                    <StatCard label="API URL" value={API_BASE_URL} accent="accent-amber" />
                  </div>
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
                displaySpecId={selectedEntryDisplayId}
                runState={selectedRunState}
                onUpdateTestCase={handleUpdateTestCase}
                onResetTestCase={handleResetTestCase}
                onRunTests={handleRunTests}
                onExplainFailure={handleExplainFailure}
                onUpdateRunConfig={handleUpdateRunConfig}
                onResetRunConfig={handleResetRunConfig}
                onApplySuggestedTest={handleApplySuggestedTest}
              />
            </>
          )}

          {showChatPanel ? (
            <ChatPanel
              selectedEntry={selectedEntry}
              specDisplayIdBySpecId={specDisplayIdBySpecId}
              messages={chatMessages}
              draft={chatDraft}
              pending={chatPending}
              onDraftChange={setChatDraft}
              onSend={handleChatSend}
            />
          ) : null}
        </main>
      )}
      <LogisticsDetailModal
        open={Boolean(logisticsDetailRunId)}
        loading={logisticsDetailLoading}
        error={logisticsDetailError}
        detail={logisticsDetail}
        deleting={Number(logisticsDeletingRunId) === Number(logisticsDetailRunId)}
        onClose={handleCloseLogisticsDetail}
        onDelete={handleDeleteLogisticsRun}
        onShowJson={handleOpenJsonViewer}
      />
      <JsonViewerModal
        open={Boolean(jsonViewerState?.open)}
        title={jsonViewerState?.title || "JSON Viewer"}
        value={jsonViewerState?.value}
        onClose={handleCloseJsonViewer}
      />
      <SettingsModal
        open={Boolean(session) && isSettingsOpen}
        tab={settingsTab}
        loading={llmSettingsLoading}
        busy={llmSettingsBusy}
        error={llmSettingsError}
        settings={llmSettings}
        isAddModelFormOpen={isAddModelFormOpen}
        addModelForm={addModelForm}
        providerModels={providerModels}
        providerModelsLoading={providerModelsLoading}
        providerModelsError={providerModelsError}
        customInstructionDraft={customInstructionDraft}
        onClose={() => {
          setIsSettingsOpen(false);
          setIsAddModelFormOpen(false);
          setAddModelForm({ ...DEFAULT_ADD_MODEL_FORM });
          setProviderModels([]);
          setProviderModelsLoading(false);
          setProviderModelsError("");
          providerModelsRequestRef.current += 1;
        }}
        onTabChange={setSettingsTab}
        onToggleAddModelForm={() => {
          setIsAddModelFormOpen((current) => {
            const next = !current;
            if (!next) {
              setAddModelForm({ ...DEFAULT_ADD_MODEL_FORM });
              setProviderModels([]);
              setProviderModelsLoading(false);
              setProviderModelsError("");
              providerModelsRequestRef.current += 1;
            }
            return next;
          });
        }}
        onAddModelFieldChange={handleAddModelFieldChange}
        onAddModelSubmit={handleAddModelSubmit}
        onSelectModel={handleSelectModel}
        onDeleteModel={handleDeleteModel}
        onCustomInstructionDraftChange={setCustomInstructionDraft}
        onSaveCustomInstruction={handleSaveCustomInstruction}
        onResetCustomInstruction={handleResetCustomInstruction}
      />
    </div>
  );
}
