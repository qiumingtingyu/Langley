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
    const source = FakeEventSource.instances[0]!;
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
    const first = FakeEventSource.instances[0]!;
    enqueue(response({ run: run(201, "RUNNING"), assistant_message: null }));
    first.onerror?.(new Event("error"));
    await settle();
    const second = FakeEventSource.instances[1]!;
    enqueue(response({ run: run(201, "RUNNING"), assistant_message: null }));
    second.onerror?.(new Event("error"));
    await settle();
    expect(FakeEventSource.instances).toHaveLength(2);
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
});
