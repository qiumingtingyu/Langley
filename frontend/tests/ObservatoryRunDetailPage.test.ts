import { DOMWrapper, flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

import ObservatoryRunDetailPage from "../src/ObservatoryRunDetailPage.vue";

const run = {
  run_id: 42,
  business_status: "CANCELLED",
  diagnostics: {
    available: true,
    observed_outcome: "FAILED",
    trace_complete: false,
    first_observed_timestamp: "2026-09-17T10:00:00Z",
    last_observed_timestamp: "2026-09-17T10:00:02Z",
    provider: "qwen",
    configured_model: "model-a",
    capture_mode: "FULL_CONTENT",
    round_count: 2,
    tool_count: 1,
    provider_input_tokens: null,
    provider_output_tokens: 30,
    provider_total_tokens: null,
    observed_duration_ms: 2000,
  },
};

const events = [
  { event_id: 1, schema_version: 1, observed_seq: 1, kind: "run.start", round: null, timestamp: "2026-09-17T10:00:00Z", duration_ms: null, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: {} },
  { event_id: 2, schema_version: 1, observed_seq: 2, kind: "llm", round: 1, timestamp: "2026-09-17T10:00:00.2Z", duration_ms: 400, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: { finish_reason: "tool_calls" } },
  { event_id: 3, schema_version: 1, observed_seq: 3, kind: "tool", round: 1, timestamp: "2026-09-17T10:00:00.7Z", duration_ms: 20, tool_call_id: "call-1", parent_tool_call_id: null, tool_ordinal: 1, metadata: { tool_name: "knowledge_search" } },
  { event_id: 4, schema_version: 1, observed_seq: 4, kind: "knowledge.search", round: 1, timestamp: "2026-09-17T10:00:00.8Z", duration_ms: 10, tool_call_id: null, parent_tool_call_id: "call-1", tool_ordinal: null, metadata: { hit_count: 3 } },
  { event_id: 5, schema_version: 1, observed_seq: 5, kind: "llm", round: 2, timestamp: "2026-09-17T10:00:01Z", duration_ms: 500, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: { finish_reason: "stop" } },
  { event_id: 6, schema_version: 1, observed_seq: 6, kind: "future.event", round: 2, timestamp: "2026-09-17T10:00:01.7Z", duration_ms: null, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: { future_flag: true } },
  { event_id: 7, schema_version: 1, observed_seq: 7, kind: "citation.validate", round: 2, timestamp: "2026-09-17T10:00:01.8Z", duration_ms: null, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: { success: true } },
  { event_id: 8, schema_version: 1, observed_seq: 8, kind: "run.failure", round: null, timestamp: "2026-09-17T10:00:02Z", duration_ms: null, tool_call_id: null, parent_tool_call_id: null, tool_ordinal: null, metadata: { error_code: "CANCELLED" } },
];

const rounds = {
  1: { run_id: 42, round: 1, provider_input_tokens: null, provider_output_tokens: 10, provider_total_tokens: null, duration_ms: 400, ttft_ms: 60, finish_reason: "tool_calls", provider_model: "model-a", estimate_kind: "conservative_multilingual_v1", estimated_semantic_context_tokens: 100 },
  2: { run_id: 42, round: 2, provider_input_tokens: 120, provider_output_tokens: 20, provider_total_tokens: 140, duration_ms: 500, ttft_ms: 75, finish_reason: "stop", provider_model: "model-a", estimate_kind: "conservative_multilingual_v1", estimated_semantic_context_tokens: 150 },
};

const contexts = {
  1: {
    run_id: 42, round: 1, estimate_kind: "conservative_multilingual_v1", estimated_semantic_context_tokens: 100,
    components: [
      { key: "system", kind: "system", source: "prompt", chars: 20, utf8_bytes: 20, estimated_tokens: 10, content_sha256: "bbb" },
      { key: "transcript.0.user", kind: "transcript.user", source: "transcript[0]", chars: 12, utf8_bytes: 12, estimated_tokens: 5, content_sha256: null },
      { key: "skill:old", kind: "skill.active", source: "skill", chars: 6, utf8_bytes: 6, estimated_tokens: 3, content_sha256: "old" },
    ],
  },
  2: {
    run_id: 42, round: 2, estimate_kind: "conservative_multilingual_v1", estimated_semantic_context_tokens: 150,
    components: [
      { key: "system", kind: "system", source: "prompt", chars: 20, utf8_bytes: 20, estimated_tokens: 10, content_sha256: "aaa" },
      { key: "personal_context.0", kind: "personal_context", source: "personal_context[0]", chars: 14, utf8_bytes: 18, estimated_tokens: 6, content_sha256: "memory" },
      { key: "conversation_compact", kind: "conversation_compact", source: null, chars: 16, utf8_bytes: 20, estimated_tokens: 7, content_sha256: "compact" },
      { key: "skill.active", kind: "skill.active", source: "active-skill", chars: 30, utf8_bytes: 30, estimated_tokens: 12, content_sha256: "skill" },
      { key: "skill.catalog.0", kind: "skill.catalog", source: "catalog-skill", chars: 26, utf8_bytes: 26, estimated_tokens: 10, content_sha256: "catalog" },
      { key: "skill.resources.0", kind: "skill.resources", source: "reference.md", chars: 25, utf8_bytes: 25, estimated_tokens: 9, content_sha256: "resource" },
      { key: "tools.schema.0", kind: "tools.schema", source: "search_knowledge", chars: 50, utf8_bytes: 50, estimated_tokens: 20, content_sha256: "tool" },
      { key: "transcript.0.user", kind: "transcript.user", source: "transcript[0]", chars: 12, utf8_bytes: 12, estimated_tokens: 5, content_sha256: "user" },
      { key: "transcript.1.assistant", kind: "transcript.assistant", source: "transcript[1]", chars: 17, utf8_bytes: 17, estimated_tokens: 7, content_sha256: "assistant" },
      { key: "transcript.1.tool_call.0", kind: "transcript.tool_call", source: "search_knowledge", chars: 60, utf8_bytes: 60, estimated_tokens: 24, content_sha256: "call" },
      { key: "transcript.2.tool_result", kind: "transcript.tool_result", source: "search_knowledge", chars: 70, utf8_bytes: 70, estimated_tokens: 28, content_sha256: "result" },
      { key: "evidence_context", kind: "evidence_context", source: null, chars: 18, utf8_bytes: 22, estimated_tokens: 8, content_sha256: "evidence-body" },
      { key: "future:0", kind: "future.context", source: null, chars: 5, utf8_bytes: 5, estimated_tokens: 2, content_sha256: null },
      { key: "unclassified.request_field.future", kind: "unclassified.request_field", source: "future", chars: 8, utf8_bytes: 8, estimated_tokens: 3, content_sha256: null },
    ],
  },
};

function response(payload: unknown, status = 200): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => payload } as Response;
}

function deferred<ResponseValue>(): { promise: Promise<ResponseValue>; resolve: (value: ResponseValue) => void } {
  let resolve!: (value: ResponseValue) => void;
  const promise = new Promise<ResponseValue>(done => {
    resolve = done;
  });
  return { promise, resolve };
}

function defaultFetch(path: string): Promise<Response> {
  if (path === "/api/dev/observatory/runs/42") return Promise.resolve(response(run));
  if (path === "/api/dev/observatory/runs/42/timeline") return Promise.resolve(response({ run_id: 42, events }));
  const roundMatch = path.match(/\/rounds\/(\d+)$/);
  if (roundMatch) return Promise.resolve(response(rounds[Number(roundMatch[1]) as 1 | 2]));
  const contextMatch = path.match(/\/rounds\/(\d+)\/context$/);
  if (contextMatch) return Promise.resolve(response(contexts[Number(contextMatch[1]) as 1 | 2]));
  if (path === "/api/dev/observatory/runs/42/events/2/raw") {
    return Promise.resolve(response({ run_id: 42, event_id: 2, capture_mode: "FULL_CONTENT", event: { kind: "llm", request: { messages: ["secret"] } } }));
  }
  if (path === "/api/dev/observatory/runs/42/events/5/raw") {
    return Promise.resolve(response({
      run_id: 42,
      event_id: 5,
      capture_mode: "FULL_CONTENT",
      event: {
        kind: "llm",
        round: 2,
        request: {
          system_input: "You are the Langley system.",
          personal_context: ["用户偏好中文回答"],
          conversation_compact_context: "此前讨论了 TCP。",
          active_skill: { name: "active-skill", instructions: "Follow the diagnostic procedure." },
          available_skills: [{ name: "catalog-skill", description: "A visible catalog entry" }],
          active_skill_resources: [{ path: "reference.md", byte_size: 17 }],
          allowed_tools: [{ name: "search_knowledge", description: "Search evidence", arguments_schema: { type: "object" } }],
          transcript: [
            { role: "user", content: "TCP 为什么需要四次挥手？" },
            { role: "assistant", content: "我先检索证据。", tool_calls: [{ call_id: "call-1", name: "search_knowledge", raw_arguments: "{\"query\":\"TCP\"}" }] },
            { call_id: "call-1", name: "search_knowledge", kind: "SUCCESS", content: "主动关闭方进入 TIME_WAIT。" },
          ],
          evidence_context: "证据指出 TCP 关闭是双向过程。",
        },
      },
    }));
  }
  throw new Error(`unexpected fetch: ${path}`);
}

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("ObservatoryRunDetailPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("promotes the CTA and loads the selected Round LLM request only after opening the Sheet", async () => {
    const fetchMock = vi.fn(defaultFetch);
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 }, attachTo: document.body });
    await settle();

    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith("/events/5/raw"))).toBe(false);
    const contextAction = wrapper.get(".context-inspector-action");
    const measurementGrid = wrapper.get(".measurement-grid");
    expect(contextAction.text()).toContain("CONTEXT INSPECTOR");
    expect(contextAction.text()).toContain("检查本轮模型上下文");
    expect(contextAction.text()).toContain("R2 · FULL_CONTENT · 约 150 Token");
    expect(
      contextAction.element.compareDocumentPosition(measurementGrid.element)
      & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    await contextAction.get("button").trigger("click");
    await settle();

    expect(fetchMock).toHaveBeenCalledWith("/api/dev/observatory/runs/42/events/5/raw");
    const dialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[data-slot="sheet-content"]')!);
    expect(dialog.text()).toContain("Round 2 上下文");
    expect(dialog.text()).toContain("模型请求上下文（provider-neutral）");
    expect(dialog.text()).toContain("Tool Schema");
    expect(dialog.text()).toContain("search_knowledge");
    expect(dialog.text()).toContain("未知上下文组件");
    expect(dialog.text()).toContain("本地 FULL_CONTENT Trace 可能包含会话、Memory、Knowledge、Tool、Skill 与 Workspace 内容。");
    expect(dialog.text()).toContain("ID #5 仅用于定位，不代表执行时间顺序");
    expect(dialog.find('[data-provenance="system"]').exists()).toBe(true);
    expect(dialog.find('[data-provenance="unknown"]').exists()).toBe(true);
    expect(dialog.findAll(".context-collapse-trigger")).toHaveLength(contexts[2].components.length);
    expect(dialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);

    await dialog.get('[data-provenance="system"] .context-collapse-trigger').trigger("click");
    await dialog.get('[data-provenance="tools"] .context-collapse-trigger').trigger("click");
    await settle();
    expect(dialog.text()).toContain("You are the Langley system.");
    expect(dialog.text()).toContain('"arguments_schema"');
    expect(dialog.findAll(".context-collapse-trigger").filter(trigger => trigger.text().includes("收起"))).toHaveLength(2);

    await dialog.get('button[aria-label="关闭面板"]').trigger("click");
    await settle();
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    await wrapper.get(".context-inspector-action button").trigger("click");
    await settle();
    const reopenedDialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(reopenedDialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);

    const rawTab = reopenedDialog.findAll('[role="tab"]').find(tab => tab.text().includes("Raw JSON"))!;
    await rawTab.trigger("mousedown", { button: 0, ctrlKey: false });
    await settle();
    expect(reopenedDialog.text()).toContain('"system_input": "You are the Langley system."');
    expect(reopenedDialog.text()).toContain('"raw_arguments": "{\\"query\\":\\"TCP\\"}"');
    expect(fetchMock.mock.calls.filter(([path]) => String(path).endsWith("/events/5/raw"))).toHaveLength(2);
    wrapper.unmount();
  });

  it("keeps ContextFrame cards useful when the explicit raw load is METADATA_ONLY", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path === "/api/dev/observatory/runs/42") {
        return Promise.resolve(response({ ...run, diagnostics: { ...run.diagnostics, capture_mode: "METADATA_ONLY" } }));
      }
      if (path === "/api/dev/observatory/runs/42/events/5/raw") {
        return Promise.resolve(response({ run_id: 42, event_id: 5, capture_mode: "METADATA_ONLY", event: { kind: "llm", round: 2 } }));
      }
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 }, attachTo: document.body });
    await settle();
    await wrapper.get(".context-inspector-action button").trigger("click");
    await settle();

    const dialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(dialog.text()).toContain("系统提示");
    expect(dialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);
    await dialog.get('[data-provenance="system"] .context-collapse-trigger').trigger("click");
    await settle();
    expect(dialog.text()).toContain("正文未捕获 · METADATA_ONLY");
    expect(dialog.text()).not.toContain("暂无上下文");
    wrapper.unmount();
  });

  it("does not guess an event ID when the selected Round has no LLM event", async () => {
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/dev/observatory/runs/42/timeline") {
        return Promise.resolve(response({ run_id: 42, events: events.filter(event => !(event.kind === "llm" && event.round === 2)) }));
      }
      return defaultFetch(path);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 }, attachTo: document.body });
    await settle();
    await wrapper.get(".context-inspector-action button").trigger("click");
    await settle();

    const dialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(dialog.text()).toContain("没有匹配的 LLM 事件");
    expect(fetchMock.mock.calls.some(([path]) => String(path).includes("/events/5/raw"))).toBe(false);
    wrapper.unmount();
  });

  it("ignores a stale Context Inspector response after switching Rounds", async () => {
    const stale = deferred<Response>();
    const latest = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/events/5/raw")) return stale.promise;
      if (path.endsWith("/events/2/raw")) return latest.promise;
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 }, attachTo: document.body });
    await settle();
    await wrapper.get(".context-inspector-action button").trigger("click");
    await wrapper.findAll(".round-tabs button")[0]!.trigger("click");
    await settle();
    await wrapper.get(".context-inspector-action button").trigger("click");
    latest.resolve(response({ run_id: 42, event_id: 2, capture_mode: "FULL_CONTENT", event: { kind: "llm", request: { system_input: "round-one-current" } } }));
    await settle();
    stale.resolve(response({ run_id: 42, event_id: 5, capture_mode: "FULL_CONTENT", event: { kind: "llm", request: { system_input: "round-two-stale" } } }));
    await settle();

    const dialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(dialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);
    await dialog.get('[data-provenance="system"] .context-collapse-trigger').trigger("click");
    await settle();
    expect(dialog.text()).toContain("round-one-current");
    expect(dialog.text()).not.toContain("round-two-stale");
    wrapper.unmount();
  });

  it("renders authority distinctions, correlated events, Round detail, ContextFrame composition, and diff", async () => {
    const fetchMock = vi.fn(defaultFetch);
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("业务状态 · MySQL 权威");
    expect(wrapper.text()).toContain("CANCELLED");
    expect(wrapper.text()).toContain("观测结果 · diagnostics");
    expect(wrapper.text()).toContain("FAILED");
    expect(wrapper.text()).toContain("Trace 完整性");
    expect(wrapper.text()).toContain("观测时长");
    expect(wrapper.text()).toContain("可能并非完整 Run 延迟");
    expect(wrapper.text()).toContain("Provider 输出");
    expect(wrapper.text()).toContain("Provider 总计");
    expect(wrapper.text()).toContain("Rounds / Tools");
    expect(wrapper.text()).toContain("run.start");
    expect(wrapper.text()).toContain("Run 根事件");
    expect(wrapper.text()).toContain("Round 1");
    expect(wrapper.text()).toContain("knowledge.search");
    expect(wrapper.text()).toContain("citation.validate");
    expect(wrapper.text()).toContain("future.event");
    expect(wrapper.text()).toContain("Round 2");
    expect(wrapper.text()).toContain("120");
    expect(wrapper.text()).toContain("conservative_multilingual_v1 · 非 Provider 精确 Token");
    expect(wrapper.text()).toContain("本轮上下文构成");
    expect(wrapper.text()).toContain("各类组件占用的语义 Token 估算");
    expect(wrapper.text()).toContain("本轮语义总估算");
    expect(wrapper.text()).toContain("组件明细");
    expect(wrapper.text()).toContain("与上一 Round 的变化");
    expect(wrapper.text()).toContain("evidence_context");
    expect(wrapper.text()).toContain("future.context");
    expect(wrapper.findAll(".table-scroll tbody tr")).toHaveLength(5);
    await wrapper.get(".component-detail-toggle button").trigger("click");
    expect(wrapper.findAll(".table-scroll tbody tr")).toHaveLength(14);
    expect(wrapper.text()).toContain("ADDED");
    expect(wrapper.text()).toContain("REMOVED");
    expect(wrapper.text()).toContain("CHANGED");
    expect(wrapper.text()).not.toContain("保留 · RETAINED");
    expect(wrapper.findAll(".diff-grid article").map(item => item.attributes("data-status"))).toEqual([
      ...Array(12).fill("ADDED"),
      "CHANGED",
      "REMOVED",
    ]);
    await wrapper.get(".retained-disclosure button").trigger("click");
    expect(wrapper.text()).toContain("保留 · RETAINED");
    expect(wrapper.text()).toContain("不代表已验证正文完全相同");
    expect(wrapper.get('[aria-label="Context 总量对比"]').text()).toContain("100");
    expect(wrapper.get('[aria-label="Context 总量对比"]').text()).toContain("150");
    expect(wrapper.get('[aria-label="Context 总量对比"]').text()).toContain("+50");
    expect(wrapper.get('[data-testid="trend-provider-input-tokens-by-round"]').find('[data-point-id="1"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="trend-provider-input-tokens-by-round"]').findAll(".x-tick-label").map(label => label.text())).toEqual(["R1", "R2"]);
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes("/events/"))).toBe(true);

    expect(wrapper.findAll("button").some(button => button.text().includes("原始记录"))).toBe(true);
    await wrapper.get('button[aria-label="查看原始诊断记录 2"]').trigger("click");
    await settle();
    expect(fetchMock).toHaveBeenCalledWith("/api/dev/observatory/runs/42/events/2/raw");
    expect(wrapper.text()).toContain("原始诊断记录 · ID #2");
    expect(wrapper.text()).toContain("事件 ID 仅用于定位，不代表时间顺序");
    expect(wrapper.text()).toContain("会话、Memory、Knowledge、Tool 或 Workspace 内容");
    expect(wrapper.text()).toContain("secret");

    await wrapper.get(".round-tabs button").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("首个已观测 Round 没有上一轮 ContextFrame 基线");
    expect(wrapper.text()).toContain("未捕获");
  });

  it("keeps overview usable when Timeline fails", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) =>
      path.endsWith("/timeline") ? Promise.resolve(response({}, 500)) : Promise.resolve(response(run)),
    ));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("CANCELLED");
    expect(wrapper.text()).toContain("执行时间线暂不可用");
    expect(wrapper.text()).toContain("无法生成 Round 趋势");
  });

  it("keeps Timeline and provider measurements usable when ContextFrame fails", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) =>
      path.endsWith("/context") ? Promise.resolve(response({}, 404)) : defaultFetch(path),
    ));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("knowledge.search");
    expect(wrapper.text()).toContain("Provider 输入");
    expect(wrapper.text()).toContain("ContextFrame 暂不可用");
  });

  it("treats a null ContextFrame projection as unavailable without fabricating composition or removals", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/rounds/2/context")) {
        return Promise.resolve(response({
          run_id: 42,
          round: 2,
          estimate_kind: null,
          estimated_semantic_context_tokens: null,
          components: [],
        }));
      }
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("Provider 输入");
    expect(wrapper.text()).toContain("120");
    expect(wrapper.text()).toContain("本轮未捕获 ContextFrame");
    expect(wrapper.text()).not.toContain("本轮上下文构成");
    expect(wrapper.text()).toContain("仍可切换到其他 Round");
    expect(wrapper.get('button[aria-label="查看上一 Round"]').attributes("disabled")).toBeUndefined();
    await wrapper.get('button[aria-label="查看上一 Round"]').trigger("click");
    await settle();
    expect(wrapper.get("#round-title").text()).toBe("Round 1");
    expect(wrapper.text()).toContain("本轮上下文构成");
    expect(wrapper.find(".diff-grid").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("REMOVED");
  });

  it("does not classify components as added when the previous ContextFrame baseline is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/rounds/1/context")) {
        return Promise.resolve(response({
          run_id: 42,
          round: 1,
          estimate_kind: null,
          estimated_semantic_context_tokens: null,
          components: [],
        }));
      }
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("本轮上下文构成");
    expect(wrapper.text()).toContain("上一轮 ContextFrame 不可用");
    expect(wrapper.find(".diff-grid").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("ADDED");
  });

  it("does not compare semantic totals or components across estimator versions", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/rounds/1/context")) {
        return Promise.resolve(response({ ...contexts[1], estimate_kind: "conservative_multilingual_v0" }));
      }
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    expect(wrapper.text()).toContain("Context 估算器版本不同");
    expect(wrapper.find('[aria-label="Context 总量对比"]').exists()).toBe(false);
    expect(wrapper.find(".diff-grid").exists()).toBe(false);
  });

  it("pages through every indexed Round with shared selection and disabled ends", async () => {
    const fetchMock = vi.fn(defaultFetch);
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 }, attachTo: document.body });
    await settle();

    const previous = wrapper.get('button[aria-label="查看上一 Round"]');
    const next = wrapper.get('button[aria-label="查看下一 Round"]');
    expect(previous.attributes("disabled")).toBeUndefined();
    expect(next.attributes("disabled")).toBeDefined();

    await previous.trigger("click");
    await settle();
    expect(wrapper.get("#round-title").text()).toBe("Round 1");
    expect(wrapper.get('button[aria-label="查看上一 Round"]').attributes("disabled")).toBeDefined();
    expect(wrapper.get('button[aria-label="查看下一 Round"]').attributes("disabled")).toBeUndefined();
    expect(wrapper.find('.timeline-group[data-selected="true"] .round-heading').text()).toContain("Round 1");

    await wrapper.get(".context-inspector-action button").trigger("click");
    await settle();
    expect(fetchMock).toHaveBeenCalledWith("/api/dev/observatory/runs/42/events/2/raw");
    const roundOneDialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(roundOneDialog.text()).toContain("Round 1 上下文");
    expect(roundOneDialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);
    await roundOneDialog.get('[data-provenance="system"] .context-collapse-trigger').trigger("click");
    await settle();
    expect(roundOneDialog.get('[data-provenance="system"] .context-collapse-trigger').text()).toContain("收起");

    await wrapper.get('button[aria-label="查看下一 Round"]').trigger("click");
    await settle();
    expect(wrapper.get("#round-title").text()).toBe("Round 2");
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    await wrapper.get(".context-inspector-action button").trigger("click");
    await settle();
    const roundTwoDialog = new DOMWrapper(document.body.querySelector<HTMLElement>('[role="dialog"]')!);
    expect(roundTwoDialog.text()).toContain("Round 2 上下文");
    expect(roundTwoDialog.findAll(".context-collapse-trigger").every(trigger => trigger.text().includes("展开"))).toBe(true);
    wrapper.unmount();
  });

  it("reports metadata-only and unavailable raw events without affecting other sections", async () => {
    let rawUnavailable = false;
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path === "/api/dev/observatory/runs/42/events/1/raw") {
        return Promise.resolve(rawUnavailable
          ? response({ detail: { code: "OBSERVATORY_RAW_EVENT_UNAVAILABLE" } }, 409)
          : response({ run_id: 42, event_id: 1, capture_mode: "METADATA_ONLY", event: { kind: "run.start" } }));
      }
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    await wrapper.get('button[aria-label="查看原始诊断记录 1"]').trigger("click");
    await settle();
    expect(wrapper.text()).toContain("当前为 METADATA_ONLY");
    expect(wrapper.text()).toContain("run.start");

    rawUnavailable = true;
    await wrapper.get('button[aria-label="查看原始诊断记录 1"]').trigger("click");
    await settle();
    expect(wrapper.text()).toContain("该 Trace 的原始诊断记录不可用");
    expect(wrapper.text()).toContain("Run 概览");
  });

  it("keeps the latest raw payload when an older successful request resolves last", async () => {
    const first = deferred<Response>();
    const second = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/events/1/raw")) return first.promise;
      if (path.endsWith("/events/2/raw")) return second.promise;
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    await wrapper.get('button[aria-label="查看原始诊断记录 1"]').trigger("click");
    await wrapper.get('button[aria-label="查看原始诊断记录 2"]').trigger("click");
    second.resolve(response({ run_id: 42, event_id: 2, capture_mode: "FULL_CONTENT", event: { marker: "event-two" } }));
    await settle();
    first.resolve(response({ run_id: 42, event_id: 1, capture_mode: "FULL_CONTENT", event: { marker: "event-one" } }));
    await settle();

    expect(wrapper.text()).toContain("原始诊断记录 · ID #2");
    expect(wrapper.text()).toContain("event-two");
    expect(wrapper.text()).not.toContain("event-one");
  });

  it("ignores stale raw errors and loading cleanup after the latest request succeeds", async () => {
    const first = deferred<Response>();
    const second = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn((path: string) => {
      if (path.endsWith("/events/1/raw")) return first.promise;
      if (path.endsWith("/events/2/raw")) return second.promise;
      return defaultFetch(path);
    }));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();

    await wrapper.get('button[aria-label="查看原始诊断记录 1"]').trigger("click");
    await wrapper.get('button[aria-label="查看原始诊断记录 2"]').trigger("click");
    second.resolve(response({ run_id: 42, event_id: 2, capture_mode: "FULL_CONTENT", event: { marker: "latest" } }));
    await settle();
    first.resolve(response({ detail: { code: "OBSERVATORY_RAW_EVENT_UNAVAILABLE" } }, 409));
    await settle();

    expect(wrapper.text()).toContain("原始诊断记录 · ID #2");
    expect(wrapper.text()).toContain("latest");
    expect(wrapper.text()).not.toContain("该 Trace 的原始诊断记录不可用");
    expect(wrapper.text()).not.toContain("正在加载原始诊断记录");
  });

  it("emits back without mutating the inspected Run", async () => {
    vi.stubGlobal("fetch", vi.fn(defaultFetch));
    const wrapper = mount(ObservatoryRunDetailPage, { props: { runId: 42 } });
    await settle();
    await wrapper.get("header button").trigger("click");
    expect(wrapper.emitted("back")).toHaveLength(1);
  });
});
