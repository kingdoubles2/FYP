const MOCK_RESPONSE_DELAY_MS = 640;

function createMessageId(prefix = "chat") {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeContext(context) {
  const mode = context?.mode === "spec" ? "spec" : "global";
  if (mode !== "spec") {
    return { mode: "global", specId: null };
  }

  const specId = Number(context?.specId);
  if (!Number.isFinite(specId)) {
    return { mode: "global", specId: null };
  }

  return { mode: "spec", specId };
}

function normalizeRuntimeConfig(runtimeConfig) {
  return {
    modelId: String(runtimeConfig?.modelId || "qwen3-coder:latest"),
    userInstruction: String(runtimeConfig?.userInstruction || ""),
  };
}

function buildMockContent({ message, context, runtimeConfig }) {
  const scopeLine = context.mode === "spec" && context.specId !== null
    ? `Scope: Selected spec #${context.specId}.`
    : "Scope: Global.";
  const instructionLine = runtimeConfig.userInstruction.trim()
    ? `User instruction preset: "${runtimeConfig.userInstruction.trim()}".`
    : "User instruction preset: (none configured in UI yet).";

  return [
    "Mock adapter response.",
    scopeLine,
    instructionLine,
    `Model target (hidden runtime config): ${runtimeConfig.modelId}.`,
    "",
    `Echo: ${message}`,
  ].join("\n");
}

export const mockChatAdapter = {
  async send(request) {
    const message = String(request?.message || "").trim();
    if (!message) {
      throw new Error("Cannot send an empty message.");
    }

    const context = normalizeContext(request?.context);
    const runtimeConfig = normalizeRuntimeConfig(request?.runtimeConfig);
    const startedAt = Date.now();
    await new Promise((resolve) => {
      window.setTimeout(resolve, MOCK_RESPONSE_DELAY_MS);
    });
    const latencyMs = Date.now() - startedAt;

    return {
      assistantMessage: {
        id: createMessageId("assistant"),
        role: "assistant",
        content: buildMockContent({ message, context, runtimeConfig }),
        createdAt: new Date().toISOString(),
        context,
        meta: {
          adapter: "mock",
          modelId: runtimeConfig.modelId,
          latencyMs,
        },
      },
      meta: {
        adapter: "mock",
        modelId: runtimeConfig.modelId,
        latencyMs,
      },
    };
  },
};
