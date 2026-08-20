import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../src/App.vue";

type RunStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";
type Run = { id: number; input_message_id: number; attempt_no: number; status: RunStatus; started_at: string | null; finished_at: string | null; error_code: string | null };
type Message = { id: number; sequence_no: number; role: "USER" | "ASSISTANT"; content: string; run_id: number | null; regenerated_from_message_id: number | null; created_at: string };

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly listeners = new Map<string, EventListener[]>();
  closed = false;
  onerror: ((event: Event) => void) | null = null;

  constructor(readonly url: string) { FakeEventSource.instances.push(this); }
  addEventListener(type: string, listener: EventListener): void { this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]); }
  close(): void { this.closed = true; }
  emit(type: string, event = new Event(type)): void { for (const listener of this.listeners.get(type) ?? []) listener(event); }
}

function deferred<Value>(): { promise: Promise<Value>; resolve(value: Value): void } {
  let resolve: (value: Value) => void;
  const promise = new Promise<Value>((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve: resolve! };
}

function response(payload: unknown, ok = true): Response { return { ok, json: async () => payload } as Response; }
function conversation(id: number, title: string) { return { id, title, created_at: "2026-08-16T00:00:00Z", updated_at: "2026-08-16T00:00:00Z", last_message_at: null }; }
function message(id: number, content: string, role: Message["role"] = "USER", runId: number | null = null): Message { return { id, sequence_no: id, role, content, run_id: runId, regenerated_from_message_id: null, created_at: "2026-08-16T00:00:00Z" }; }
function run(id: number, status: RunStatus): Run { return { id, input_message_id: id, attempt_no: 1, status, started_at: status === "PENDING" ? null : "2026-08-16T00:00:00Z", finished_at: ["SUCCEEDED", "FAILED", "CANCELLED"].includes(status) ? "2026-08-16T00:01:00Z" : null, error_code: status === "FAILED" ? "LLM_PROVIDER_FAILED" : null }; }
function button(wrapper: VueWrapper, text: string) { const found = wrapper.findAll("button").find((candidate) => candidate.text().includes(text)); if (found === undefined) throw new Error(`button not found: ${text}`); return found; }
async function settle(): Promise<void> { await flushPromises(); await nextTick(); await flushPromises(); }

describe("App user behavior", () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  let enqueue: (value: Response | Promise<Response>) => void;

  beforeEach(() => {
    const responses: Array<Response | Promise<Response>> = [];
    fetchMock = vi.fn(() => { const next = responses.shift(); if (next === undefined) throw new Error("unexpected fetch"); return Promise.resolve(next); });
    enqueue = (value) => responses.push(value);
    FakeEventSource.instances = [];
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("EventSource", FakeEventSource);
    let request = 0;
    vi.stubGlobal("crypto", { randomUUID: () => `request-${++request}` });
  });

  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  async function mountInitial(activeRun: Run | null = null): Promise<VueWrapper> {
    enqueue(response([conversation(1, "A"), conversation(2, "B")]));
    enqueue(response({ messages: [message(1, "问题 A")], latest_run: activeRun }));
    const wrapper = mount(App);
    await settle();
    return wrapper;
  }

  async function sendQuestion(wrapper: VueWrapper, content: string): Promise<void> {
    await wrapper.get("textarea").setValue(content);
    await wrapper.get("form").trigger("submit");
    await nextTick();
  }

  it("does not let an old A view overwrite the newer A view after A to B to A", async () => {
    const wrapper = await mountInitial();
    const oldADetail = deferred<Response>();
    let aDetailRequests = 0;
    fetchMock.mockImplementation((path: string) => {
      if (path === "/api/conversations") {
        return Promise.resolve(response([conversation(1, "A"), conversation(2, "B")]));
      }
      if (path === "/api/conversations/1/messages") {
        aDetailRequests += 1;
        return Promise.resolve(
          aDetailRequests === 1
            ? oldADetail.promise
            : response({ messages: [message(3, "新的 A")], latest_run: null }),
        );
      }
      if (path === "/api/conversations/2/messages") {
        return Promise.resolve(response({ messages: [message(2, "B 的内容")], latest_run: null }));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    await wrapper.get('button[aria-label="刷新会话"]').trigger("click");
    await vi.waitFor(() => expect(aDetailRequests).toBe(1));

    await button(wrapper, "B").trigger("click");
    await settle();

    await button(wrapper, "A").trigger("click");
    await settle();

    oldADetail.resolve(response({ messages: [message(4, "过期 A")], latest_run: null }));
    await settle();

    expect(wrapper.find("header").text()).toContain("A");
    expect(wrapper.text()).toContain("新的 A");
    expect(wrapper.text()).not.toContain("过期 A");
    wrapper.unmount();
  });

  it("keeps the client request id when a network command is retried", async () => {
    const wrapper = await mountInitial();
    enqueue(Promise.reject(new TypeError("network failure")));
    await sendQuestion(wrapper, "第一个问题");
    await settle();
    expect(button(wrapper, "重试请求").exists()).toBe(true);
    enqueue(response({ user_message: message(2, "第一个问题"), run: run(201, "PENDING") }));
    await button(wrapper, "重试请求").trigger("click");
    await settle();
    const requests = fetchMock.mock.calls.filter(([path, init]) => path === "/api/conversations/1/messages" && init?.method === "POST");
    expect(requests).toHaveLength(2);
    expect(JSON.parse(String(requests[0]?.[1]?.body)).client_request_id).toBe(JSON.parse(String(requests[1]?.[1]?.body)).client_request_id);
    wrapper.unmount();
  });

  it("streams a delta and reads the durable assistant after success", async () => {
    const wrapper = await mountInitial(run(201, "RUNNING"));
    const source = FakeEventSource.instances.find((item) => item.url.includes("/api/runs/201/events"))!;
    source.emit("message.delta", new MessageEvent("message.delta", { data: JSON.stringify({ run_id: 201, delta: "流式内容" }) }));
    await settle();
    expect(wrapper.text()).toContain("流式内容");
    enqueue(response({ run: run(201, "SUCCEEDED"), assistant_message: message(2, "已保存回答", "ASSISTANT", 201) }));
    enqueue(response({ messages: [message(1, "问题 A"), message(2, "已保存回答", "ASSISTANT", 201)], latest_run: run(201, "SUCCEEDED") }));
    source.emit("run.succeeded");
    await settle();
    expect(wrapper.text()).toContain("已保存回答");
    expect(wrapper.text()).toContain("回答已保存");
    wrapper.unmount();
  });

  it("reconnects an active SSE stream only once", async () => {
    const wrapper = await mountInitial(run(201, "RUNNING"));
    const first = FakeEventSource.instances.find((item) => item.url.includes("/api/runs/201/events"))!;
    enqueue(response({ run: run(201, "RUNNING"), assistant_message: null }));
    first.onerror?.(new Event("error"));
    await settle();
    const second = FakeEventSource.instances.filter((item) => item.url.includes("/api/runs/201/events"))[1]!;
    enqueue(response({ run: run(201, "RUNNING"), assistant_message: null }));
    second.onerror?.(new Event("error"));
    await settle();
    expect(FakeEventSource.instances.filter((item) => item.url.includes("/api/runs/201/events"))).toHaveLength(2);
    expect(wrapper.get('[role="alert"]').text()).toContain("实时连接已断开");
    wrapper.unmount();
  });

  it("Stop reads the durable cancelled Run and does not persist an assistant", async () => {
    const wrapper = await mountInitial(run(201, "RUNNING"));
    enqueue(response(run(201, "CANCELLED")));
    enqueue(response({ run: run(201, "CANCELLED"), assistant_message: null }));
    enqueue(response({ messages: [message(1, "问题 A")], latest_run: run(201, "CANCELLED") }));
    await button(wrapper, "停止").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("已停止回答");
    expect(wrapper.text()).not.toContain("已保存回答");
    wrapper.unmount();
  });

  it("renames and deletes a selected conversation through the Chinese controls", async () => {
    vi.stubGlobal("prompt", () => "重命名后的会话");
    vi.stubGlobal("confirm", () => true);
    const wrapper = await mountInitial();
    enqueue(response(conversation(1, "重命名后的会话")));
    await wrapper.get('button[aria-label="重命名会话"]').trigger("click");
    await settle();
    expect(wrapper.find("header").text()).toContain("重命名后的会话");
    enqueue(response({}));
    enqueue(response({ messages: [], latest_run: null }));
    await wrapper.get('button[aria-label="删除会话"]').trigger("click");
    await settle();
    expect(wrapper.find("header").text()).toContain("B");
    wrapper.unmount();
  });

  it("selects a newly created conversation after the last conversation was deleted", async () => {
    vi.stubGlobal("confirm", () => true);
    let hasInitialConversation = true;
    let createdConversation = false;
    fetchMock.mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/conversations" && init?.method === "POST") {
        createdConversation = true;
        return Promise.resolve(response(conversation(2, "新会话")));
      }
      if (path === "/api/conversations") {
        return Promise.resolve(
          response(createdConversation ? [conversation(2, "新会话")] : hasInitialConversation ? [conversation(1, "唯一会话")] : []),
        );
      }
      if (path === "/api/conversations/1" && init?.method === "DELETE") {
        hasInitialConversation = false;
        return Promise.resolve(response({}));
      }
      if (path === "/api/conversations/1/messages" || path === "/api/conversations/2/messages") {
        return Promise.resolve(response({ messages: [], latest_run: null }));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    const wrapper = mount(App);
    await settle();

    await wrapper.get('button[aria-label="删除会话"]').trigger("click");
    await settle();
    expect(wrapper.find("header").text()).toContain("Langley");

    await button(wrapper, "新建会话").trigger("click");
    await flushPromises();
    await settle();

    expect(wrapper.find("header").text()).toContain("新会话");
    expect(wrapper.text()).toContain("这个会话已准备好");
    wrapper.unmount();
  });

  it("switches to Memory, loads effective facts, and serializes a local expiry absolutely", async () => {
    const wrapper = await mountInitial();
    enqueue(response({ auto_memory_enabled: true }));
    enqueue(response([]));
    await button(wrapper, "记忆").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("还没有长期记忆");
    expect(fetchMock.mock.calls.map(([path]) => path)).toContain("/api/memory-settings");
    expect(fetchMock.mock.calls.map(([path]) => path)).toContain("/api/memories");

    await wrapper.get('textarea[maxlength="1000"]').setValue("记住偏好");
    await wrapper.get('input[type="datetime-local"]').setValue("2030-01-02T03:04");
    enqueue(response({ id: 3 }));
    enqueue(response({ auto_memory_enabled: true }));
    enqueue(response([]));
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await settle();
    const add = fetchMock.mock.calls.find(([path, init]) => path === "/api/memories" && init?.method === "POST");
    expect(JSON.parse(String(add?.[1]?.body)).valid_until).toMatch(/Z$/);
    wrapper.unmount();
  });

  it("maps Memory errors, handles direct and conversational source lazily, and forgets idempotently", async () => {
    vi.stubGlobal("confirm", () => true);
    const wrapper = await mountInitial();
    enqueue(response({ auto_memory_enabled: false }));
    enqueue(response([{ id: 4, content: "直接记忆", valid_until: null, source_message_id: null, created_at: "x", updated_at: "x" }]));
    await button(wrapper, "记忆").trigger("click");
    await settle();
    expect(fetchMock.mock.calls.filter(([path]) => String(path).includes("/source"))).toHaveLength(0);
    enqueue(response({ kind: "direct" }));
    await button(wrapper, "来源").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("由你直接设置或修改");
    enqueue(response({})); enqueue(response({ auto_memory_enabled: false })); enqueue(response([]));
    await button(wrapper, "忘记").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("还没有长期记忆");
    wrapper.unmount();
  });

  it("keeps one Memory SSE across view switches, maps all outcomes, refreshes Memory, and closes on unmount", async () => {
    const wrapper = await mountInitial();
    const memorySource = FakeEventSource.instances.find((item) => item.url === "/api/memory-events")!;
    enqueue(response({ auto_memory_enabled: true })); enqueue(response([]));
    await button(wrapper, "记忆").trigger("click"); await settle();
    enqueue(response({ auto_memory_enabled: true })); enqueue(response([]));
    memorySource.emit("memory.updated", new MessageEvent("memory.updated", { data: JSON.stringify({ user_requested_memory_action: true }) })); await settle();
    expect(wrapper.text()).toContain("长期记忆已更新");
    for (const [event, message] of [["memory.no_change", "本次未对长期记忆做出修改"], ["memory.retry_pending", "长期记忆同步暂时未完成"], ["memory.not_saved", "本次内容未保存为长期记忆"]] as const) {
      memorySource.emit(event); await settle(); expect(wrapper.text()).toContain(message);
    }
    await button(wrapper, "聊天").trigger("click"); await button(wrapper, "记忆").trigger("click");
    expect(FakeEventSource.instances.filter((item) => item.url === "/api/memory-events")).toHaveLength(1);
    wrapper.unmount(); expect(memorySource.closed).toBe(true);
  });

  it("corrects without changing an absolute expiry, handles toggle rollback, and renders conversational source", async () => {
    const wrapper = await mountInitial();
    const memory = { id: 9, content: "旧内容", valid_until: "2030-01-02T03:04:37Z", source_message_id: 22, created_at: "x", updated_at: "x" };
    enqueue(response({ auto_memory_enabled: false })); enqueue(response([memory]));
    await button(wrapper, "记忆").trigger("click"); await settle();
    await button(wrapper, "修改").trigger("click");
    enqueue(response(memory)); enqueue(response({ auto_memory_enabled: false })); enqueue(response([memory]));
    await wrapper.findAll("form").at(-1)!.trigger("submit"); await settle();
    const put = fetchMock.mock.calls.find(([path, init]) => path === "/api/memories/9" && init?.method === "PUT");
    expect(JSON.parse(String(put?.[1]?.body)).valid_until).toBe("2030-01-02T03:04:37.000Z");
    enqueue(response({ auto_memory_enabled: true })); await button(wrapper, "开启").trigger("click"); await settle(); expect(wrapper.text()).toContain("自动记忆：开");
    enqueue(response({ detail: { code: "MEMORY_SYNC_UNAVAILABLE" } }, false)); await button(wrapper, "关闭").trigger("click"); await settle();
    expect(wrapper.text()).toContain("自动记忆：开"); expect(wrapper.text()).toContain("长期记忆同步暂时无法完成");
    enqueue(response({ kind: "conversation", conversation_title: null, conversation_deleted: true, context_messages: [{ id: 21, role: "USER", content: "前文" }, { id: 22, role: "USER", content: "证据" }, { id: 23, role: "ASSISTANT", content: "后文" }] }));
    await button(wrapper, "来源").trigger("click"); await settle(); expect(wrapper.text()).toContain("原会话已删除"); expect(wrapper.text()).toContain("证据");
    wrapper.unmount();
  });

  it("does not let a stale source response render under a newer Memory selection", async () => {
    const wrapper = await mountInitial();
    const memories = [
      { id: 31, content: "A", valid_until: null, source_message_id: 1, created_at: "x", updated_at: "x" },
      { id: 32, content: "B", valid_until: null, source_message_id: 2, created_at: "x", updated_at: "x" },
    ];
    enqueue(response({ auto_memory_enabled: true })); enqueue(response(memories));
    await button(wrapper, "记忆").trigger("click"); await settle();
    const oldSource = deferred<Response>(); enqueue(oldSource.promise);
    await wrapper.findAll("button").filter((item) => item.text() === "来源")[0]!.trigger("click"); await nextTick();
    enqueue(response({ kind: "direct" }));
    await wrapper.findAll("button").filter((item) => item.text() === "来源")[1]!.trigger("click"); await settle();
    oldSource.resolve(response({ kind: "conversation", conversation_title: null, conversation_deleted: false, context_messages: [{ id: 1, role: "USER", content: "stale A" }] })); await settle();
    expect(wrapper.text()).toContain("由你直接设置或修改"); expect(wrapper.text()).not.toContain("stale A");
    wrapper.unmount();
  });

  it("does not let decoded stale Memory bodies overwrite a newer reload", async () => {
    const wrapper = await mountInitial();
    const memorySource = FakeEventSource.instances.find((item) => item.url === "/api/memory-events")!;
    enqueue(response({ auto_memory_enabled: false })); enqueue(response([]));
    await button(wrapper, "记忆").trigger("click"); await settle();
    const oldSettings = deferred<unknown>(); const oldMemories = deferred<unknown>();
    enqueue({ ok: true, json: () => oldSettings.promise } as Response);
    enqueue({ ok: true, json: () => oldMemories.promise } as Response);
    memorySource.emit("memory.updated", new MessageEvent("memory.updated", { data: "{}" }));
    await flushPromises();
    enqueue(response({ auto_memory_enabled: true })); enqueue(response([{ id: 50, content: "new", valid_until: null, source_message_id: null, created_at: "x", updated_at: "x" }]));
    memorySource.emit("memory.updated", new MessageEvent("memory.updated", { data: "{}" })); await settle();
    oldSettings.resolve({ auto_memory_enabled: false }); oldMemories.resolve([{ id: 51, content: "old", valid_until: null, source_message_id: null, created_at: "x", updated_at: "x" }]); await settle();
    expect(wrapper.text()).toContain("new"); expect(wrapper.text()).not.toContain("old"); expect(wrapper.text()).toContain("自动记忆：开");
    wrapper.unmount();
  });

  it("maps not-found and validation failures without rendering backend detail", async () => {
    const wrapper = await mountInitial();
    enqueue(response({ auto_memory_enabled: true })); enqueue(response([{ id: 41, content: "现有", valid_until: null, source_message_id: null, created_at: "x", updated_at: "x" }]));
    await button(wrapper, "记忆").trigger("click"); await settle();
    enqueue(response({ detail: { code: "MEMORY_NOT_FOUND", internal: "never show" } }, false));
    await button(wrapper, "来源").trigger("click"); await settle();
    expect(wrapper.text()).toContain("这条长期记忆已不存在"); expect(wrapper.text()).not.toContain("never show");
    await wrapper.get('textarea[maxlength="1000"]').setValue("bad");
    enqueue(response({ detail: { code: "VALIDATION_ERROR", internal: "never show" } }, false));
    await wrapper.findAll("form").at(-1)!.trigger("submit"); await settle();
    expect(wrapper.text()).toContain("输入内容无效，请检查后重试"); expect(wrapper.text()).not.toContain("never show");
    wrapper.unmount();
  });
});
