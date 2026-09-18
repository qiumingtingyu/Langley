import { DOMWrapper, flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import MemoryPage from "../src/MemoryPage.vue";
import type { Memory } from "../src/types";

function response(payload: unknown, ok = true): Response {
  return { ok, json: async () => payload } as Response;
}

function status(autoMemoryEnabled: boolean, pendingEvidenceCount = 0, policyStatus = "READY") {
  return {
    auto_memory_enabled: autoMemoryEnabled,
    policy_status: policyStatus,
    pending_evidence_count: pendingEvidenceCount,
    oldest_pending_message_id: pendingEvidenceCount ? 7 : null,
    oldest_pending_created_at: pendingEvidenceCount ? "2026-08-29T01:02:03Z" : null,
  };
}

function memory(id: number, content: string): Memory {
  return { id, content, valid_until: null, source_message_id: null, created_at: "x", updated_at: "x" };
}

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("MemoryPage", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.stubGlobal("confirm", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    document.body.innerHTML = "";
  });

  it("reads operational status and effective Memory facts without exposing internal status values", async () => {
    fetchMock = vi.fn((path: string) => {
      if (path === "/api/memory-status") return Promise.resolve(response(status(true, 2)));
      if (path === "/api/memories") return Promise.resolve(response([memory(1, "喜欢乌龙茶")]));
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(MemoryPage);
    expect(wrapper.text()).toContain("正在读取状态…");
    expect(wrapper.findAll("button").find((item) => item.text() === "正在读取状态…")?.attributes("disabled")).toBeDefined();
    await settle();

    expect(wrapper.text()).toContain("关闭自动整理");
    expect(wrapper.get("header").text()).toContain("1 条内容 · 自动整理已开启");
    expect(wrapper.text()).toContain("整理服务可用");
    expect(wrapper.text()).toContain("还有 2 条内容待整理");
    expect(wrapper.text()).toContain("喜欢乌龙茶");
    expect(wrapper.text()).not.toContain("READY");
    expect(wrapper.text()).not.toContain("oldest_pending_message_id");
    wrapper.unmount();
  });

  it("rereads durable facts after a failed sync before showing the original command error", async () => {
    let statusReads = 0;
    let memoryReads = 0;
    fetchMock = vi.fn((path: string, init?: RequestInit) => {
      if (path === "/api/memory-status") {
        statusReads += 1;
        return Promise.resolve(response(status(true, statusReads === 1 ? 3 : 1)));
      }
      if (path === "/api/memories") {
        memoryReads += 1;
        return Promise.resolve(response([memory(1, memoryReads === 1 ? "旧的长期信息" : "reread 后的长期信息")]));
      }
      if (path === "/api/memory-sync" && init?.method === "POST") {
        return Promise.resolve(response({ detail: { code: "MEMORY_SYNC_UNAVAILABLE" } }, false));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(MemoryPage);
    await settle();
    await wrapper.findAll("button").find((item) => item.text() === "立即整理")!.trigger("click");
    await settle();

    expect(wrapper.text()).toContain("还有 1 条内容待整理");
    expect(wrapper.text()).toContain("reread 后的长期信息");
    expect(wrapper.text()).not.toContain("旧的长期信息");
    expect(wrapper.emitted("notice")?.flat()).toContain("自动整理暂时不可用，请稍后重试。");
    expect(statusReads).toBe(2);
    expect(memoryReads).toBe(2);
    wrapper.unmount();
  });

  it("explains a successful bounded sync with remaining work and rereads durable state", async () => {
    let statusReads = 0;
    fetchMock = vi.fn((path: string, init?: RequestInit) => {
      if (path === "/api/memory-status") {
        statusReads += 1;
        return Promise.resolve(response(status(true, statusReads === 1 ? 7 : 3)));
      }
      if (path === "/api/memories") return Promise.resolve(response([]));
      if (path === "/api/memory-sync" && init?.method === "POST") {
        return Promise.resolve(response({ processed_count: 4, remaining_count: 3, complete: false, stop_reason: "LIMIT_REACHED", oldest_pending_message_id: 8, oldest_pending_created_at: "2026-08-29T02:00:00Z" }));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(MemoryPage);
    await settle();
    await wrapper.findAll("button").find((item) => item.text() === "立即整理")!.trigger("click");
    await settle();

    expect(wrapper.text()).toContain("本次整理了 4 条，完成时还有 3 条待整理。");
    expect(wrapper.text()).not.toContain("已整理完成。");
    expect(wrapper.text()).toContain("还有 3 条内容待整理");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memory-status")).toHaveLength(2);
    wrapper.unmount();
  });

  it("does not optimistically enable automatic organization when the command fails", async () => {
    fetchMock = vi.fn((path: string, init?: RequestInit) => {
      if (path === "/api/memory-status") return Promise.resolve(response(status(false)));
      if (path === "/api/memories") return Promise.resolve(response([]));
      if (path === "/api/memory-settings" && init?.method === "PATCH") {
        return Promise.resolve(response({ detail: { code: "MEMORY_SYNC_UNAVAILABLE" } }, false));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(MemoryPage);
    await settle();
    await wrapper.findAll("button").find((item) => item.text() === "开启自动整理")!.trigger("click");
    await settle();

    const patch = fetchMock.mock.calls.find(([path, init]) => path === "/api/memory-settings" && init?.method === "PATCH");
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ auto_memory_enabled: true });
    expect(wrapper.text()).toContain("开启自动整理");
    expect(wrapper.emitted("notice")?.flat()).toContain("自动整理暂时不可用，请稍后重试。");
    wrapper.unmount();
  });

  it("reloads facts and status after a manual write instead of patching the local list", async () => {
    let statusReads = 0;
    fetchMock = vi.fn((path: string, init?: RequestInit) => {
      if (path === "/api/memory-status") {
        statusReads += 1;
        return Promise.resolve(response(status(true)));
      }
      if (path === "/api/memories" && init?.method === "POST") return Promise.resolve(response(memory(9, "命令响应不应成为列表权威")));
      if (path === "/api/memories") return Promise.resolve(response(statusReads === 1 ? [] : [memory(9, "新的长期信息")]));
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(MemoryPage);
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    await wrapper.findAll("button").find((item) => item.text() === "添加记忆")!.trigger("click");
    await settle();
    const editor = new DOMWrapper(document.querySelector<HTMLElement>('[role="dialog"]')!);
    await editor.get('textarea[maxlength="1000"]').setValue("新的长期信息");
    await editor.get("form").trigger("submit");
    await settle();

    expect(wrapper.text()).toContain("新的长期信息");
    expect(wrapper.text()).not.toContain("命令响应不应成为列表权威");
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(JSON.parse(String(fetchMock.mock.calls.find(([, init]) => init?.method === "POST")?.[1]?.body))).toEqual({ content: "新的长期信息", valid_until: null });
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memory-status")).toHaveLength(2);
    wrapper.unmount();
  });
});


describe("Memory content-first interactions", () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); document.body.innerHTML = ""; });

  function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => { resolve = done; });
    return { promise, resolve };
  }

  function dialog() {
    const element = document.querySelector<HTMLElement>('[role="dialog"]');
    expect(element).not.toBeNull();
    return new DOMWrapper(element!);
  }

  async function setup(initial = [memory(1, "喜欢乌龙茶")], initialStatus = status(true)) {
    const facts = { memories: initial, status: initialStatus };
    const command = vi.fn<(path: string, init: RequestInit) => Promise<Response>>().mockResolvedValue(response({}));
    const sources = new Map<string, Response | Promise<Response>>();
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method) return command(path, init);
      if (path === "/api/memory-status") return response(facts.status);
      if (path === "/api/memories") return response(facts.memories);
      const sourceResponse = sources.get(path);
      if (sourceResponse) return sourceResponse;
      throw new Error("Unexpected fixture request");
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(MemoryPage, { attachTo: document.body });
    await settle();
    return { wrapper, facts, command, fetchMock, sources };
  }

  async function click(wrapper: ReturnType<typeof mount> | DOMWrapper<Element>, label: string) {
    const button = wrapper.findAll("button").find((item) => item.text() === label);
    expect(button).toBeDefined();
    await button!.trigger("click");
    await settle();
    return button!;
  }

  it("opens and cancels an empty editor without changing the content list", async () => {
    const { wrapper, command } = await setup();
    expect(wrapper.get("header").text()).toContain("1 条内容 · 自动整理已开启");
    expect(wrapper.find("textarea").exists()).toBe(false);
    const trigger = await click(wrapper, "添加记忆");
    expect(dialog().text()).toContain("添加长期信息");
    expect((dialog().get("textarea").element as HTMLTextAreaElement).value).toBe("");
    await dialog().get("textarea").setValue("Unsaved draft");
    await click(dialog(), "取消");
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger.element);
    expect(wrapper.text()).toContain("喜欢乌龙茶");
    expect(command).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("prefills edit content/expiry and PUTs the same instant before durable reread", async () => {
    const expiry = "2027-01-02T03:04:05Z";
    const record = { ...memory(1, "喜欢乌龙茶"), valid_until: expiry };
    const { wrapper, facts, command, fetchMock } = await setup([record]);
    await click(wrapper, "修改");
    expect(dialog().text()).toContain("修改长期信息");
    const editor = dialog();
    expect((editor.get("textarea").element as HTMLTextAreaElement).value).toBe(record.content);
    const localExpiry = (editor.get('input[type="datetime-local"]').element as HTMLInputElement).value;
    expect(new Date(localExpiry).toISOString()).toBe(new Date(expiry).toISOString());
    await editor.get("textarea").setValue("  新的偏好  ");
    facts.memories = [memory(1, "durable reread")];
    command.mockResolvedValue(response(memory(1, "not authority")));
    await editor.get("form").trigger("submit");
    await settle();
    expect(command).toHaveBeenCalledTimes(1);
    expect(command.mock.calls[0]![0]).toBe("/api/memories/1");
    expect(command.mock.calls[0]![1].method).toBe("PUT");
    expect(JSON.parse(String(command.mock.calls[0]![1].body))).toEqual({ content: "新的偏好", valid_until: new Date(expiry).toISOString() });
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memories")).toHaveLength(2);
    expect(wrapper.text()).toContain("durable reread");
    expect(wrapper.text()).not.toContain("not authority");
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    wrapper.unmount();
  });

  it.each(["create", "edit"] as const)("keeps the %s draft open after failure and rereads durable facts", async (mode) => {
    const { wrapper, command, fetchMock } = await setup();
    await click(wrapper, mode === "create" ? "添加记忆" : "修改");
    await dialog().get("textarea").setValue("Retry this draft");
    await dialog().get('input[type="datetime-local"]').setValue("2027-03-04T05:06:07");
    const draftExpiry = (dialog().get("input").element as HTMLInputElement).value;
    command.mockResolvedValue(response({ detail: { code: "VALIDATION_ERROR" } }, false));
    await dialog().get("form").trigger("submit");
    await settle();
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memory-status")).toHaveLength(2);
    expect((dialog().get("textarea").element as HTMLTextAreaElement).value).toBe("Retry this draft");
    expect((dialog().get("input").element as HTMLInputElement).value).toBe(draftExpiry);
    expect(dialog().get('button[type="submit"]').attributes("disabled")).toBeUndefined();
    expect(dialog().get('[role="alert"]').text()).toBe("输入内容无效，请检查后重试。");
    expect(wrapper.emitted("notice")?.flat()).toContain("输入内容无效，请检查后重试。");
    wrapper.unmount();
  });

  it("exits editing safely when failure reread finds that the target disappeared", async () => {
    const { wrapper, facts, command } = await setup();
    await click(wrapper, "修改");
    facts.memories = [];
    command.mockResolvedValue(response({ detail: { code: "MEMORY_NOT_FOUND" } }, false));
    await dialog().get("form").trigger("submit");
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(wrapper.text()).toContain("还没有长期记忆");
    await click(wrapper, "添加记忆");
    expect(dialog().text()).toContain("添加长期信息");
    expect((dialog().get("textarea").element as HTMLTextAreaElement).value).toBe("");
    wrapper.unmount();
  });

  it("keeps a pending save single-flight and does not close its editor", async () => {
    const { wrapper, command } = await setup();
    const pending = deferred<Response>();
    command.mockReturnValue(pending.promise);
    await click(wrapper, "修改");
    const editor = dialog();
    await editor.get("form").trigger("submit");
    await editor.get("form").trigger("submit");
    await editor.get('button[aria-label="关闭面板"]').trigger("click");
    await settle();
    expect(dialog().text()).toContain("正在保存…");
    expect(command).toHaveBeenCalledTimes(1);
    pending.resolve(response({}));
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    wrapper.unmount();
  });

  it("cancels forgetting with Escape and restores focus without window.confirm or DELETE", async () => {
    const confirm = vi.fn();
    vi.stubGlobal("confirm", confirm);
    const { wrapper, command } = await setup();
    const trigger = await click(wrapper, "忘记");
    expect(dialog().text()).toContain("忘记这条长期记忆？");
    expect(dialog().text()).toContain("这条信息将不再用于未来对话。");
    expect(command).not.toHaveBeenCalled();
    document.activeElement?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger.element);
    expect(confirm).not.toHaveBeenCalled();
    expect(command).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it.each(["success", "missing"] as const)("confirms forgetting once and rereads after %s", async (outcome) => {
    const { wrapper, facts, command, fetchMock } = await setup();
    const pending = deferred<Response>();
    command.mockReturnValue(pending.promise);
    await click(wrapper, "忘记");
    expect(command).not.toHaveBeenCalled();
    const confirm = dialog().findAll("button").find((item) => item.text() === "忘记")!;
    await confirm.trigger("click");
    await confirm.trigger("click");
    expect(command).toHaveBeenCalledTimes(1);
    expect(command.mock.calls[0]![0]).toBe("/api/memories/1");
    expect(command.mock.calls[0]![1].method).toBe("DELETE");
    expect(dialog().text()).toContain("正在忘记…");
    facts.memories = [];
    pending.resolve(outcome === "success" ? response({}) : response({ detail: { code: "MEMORY_NOT_FOUND" } }, false));
    await settle();
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memories")).toHaveLength(2);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(wrapper.text()).toContain("还没有长期记忆");
    if (outcome === "missing") expect(wrapper.emitted("notice")?.flat()).toContain("这条长期记忆已不存在。");
    wrapper.unmount();
  });

  it("keeps pending work visible when policy is unavailable and disables sync", async () => {
    const { wrapper, command } = await setup([], status(false, 3, "UNAVAILABLE"));
    expect(wrapper.get("header").text()).toContain("0 条内容 · 自动整理已关闭");
    expect(wrapper.text()).toContain("自动整理暂不可用");
    expect(wrapper.get('[role="status"]').text()).toContain("还有 3 条内容待整理");
    const syncButton = wrapper.findAll("button").find((item) => item.text() === "立即整理")!;
    expect(syncButton.attributes("disabled")).toBeDefined();
    await syncButton.trigger("click");
    expect(command).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("你明确提出的记住、修改或忘记请求仍可被处理");
    wrapper.unmount();
  });

  it("keeps a failed forget dialog retryable with visible feedback after durable reread", async () => {
    const { wrapper, command, fetchMock } = await setup();
    command.mockResolvedValue(response({ detail: { code: "VALIDATION_ERROR" } }, false));
    await click(wrapper, "忘记");
    await click(dialog(), "忘记");
    expect(dialog().get('[role="alert"]').text()).toBe("输入内容无效，请检查后重试。");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memories")).toHaveLength(2);
    expect(wrapper.text()).toContain("喜欢乌龙茶");
    expect(dialog().findAll("button").find((item) => item.text() === "忘记")!.attributes("disabled")).toBeUndefined();
    wrapper.unmount();
  });

  it.each(["toggle", "sync"] as const)("prevents duplicate %s requests while pending", async (operation) => {
    const { wrapper, command, fetchMock } = await setup([], status(true, 2));
    const pending = deferred<Response>();
    command.mockReturnValue(pending.promise);
    const action = wrapper.findAll("button").find((item) => item.text() === (operation === "toggle" ? "关闭自动整理" : "立即整理"))!;
    await action.trigger("click");
    await action.trigger("click");
    expect(command).toHaveBeenCalledTimes(1);
    expect(action.attributes("disabled")).toBeDefined();
    pending.resolve(response({ detail: { code: "MEMORY_SYNC_UNAVAILABLE" } }, false));
    await settle();
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memory-status")).toHaveLength(2);
    expect(wrapper.text()).toContain("自动整理已开启");
    wrapper.unmount();
  });

  it("preserves direct and deleted-conversation source display without stale source overwrite", async () => {
    const records = [memory(1, "Direct"), { ...memory(2, "Conversation"), source_message_id: 7 }];
    const { wrapper, sources } = await setup(records);
    const oldSource = deferred<Response>();
    sources.set("/api/memories/1/source", oldSource.promise);
    sources.set("/api/memories/2/source", response({ kind: "conversation", conversation_title: "Old chat", conversation_deleted: true, context_messages: [{ id: 7, role: "USER", content: "Relevant source" }, { id: 8, role: "ASSISTANT", content: "Context" }] }));
    const actions = wrapper.findAll("button").filter((item) => item.text() === "查看来源");
    await actions[0]!.trigger("click");
    await actions[1]!.trigger("click");
    await settle();
    expect(wrapper.text()).toContain("原会话已删除。");
    expect(wrapper.text()).toContain("相关对话内容");
    expect(wrapper.text()).toContain("Relevant source");
    oldSource.resolve(response({ kind: "direct" }));
    await settle();
    expect(wrapper.text()).not.toContain("这条记忆由你直接添加或修改。");
    sources.set("/api/memories/1/source", response({ kind: "direct" }));
    await actions[0]!.trigger("click");
    await settle();
    expect(wrapper.text()).toContain("这条记忆由你直接添加或修改。");
    expect(wrapper.text()).not.toContain("Relevant source");
    wrapper.unmount();
  });

  it("does not let older response bodies overwrite a newer durable load", async () => {
    const { wrapper, fetchMock } = await setup();
    const delayed = deferred<ReturnType<typeof memory>[]>();
    fetchMock.mockImplementationOnce(async () => response(status(false)))
      .mockImplementationOnce(async () => ({ ok: true, json: () => delayed.promise } as Response));
    const oldLoad = (wrapper.vm as unknown as { load(): Promise<void> }).load();
    await settle();
    await (wrapper.vm as unknown as { load(): Promise<void> }).load();
    delayed.resolve([memory(2, "Stale content")]);
    await oldLoad;
    await settle();
    expect(wrapper.text()).not.toContain("Stale content");
    expect(wrapper.text()).toContain("喜欢乌龙茶");
    expect(wrapper.get("header").text()).toContain("自动整理已开启");
    wrapper.unmount();
  });

  it("sanitizes malformed server errors while keeping draft and failure reread", async () => {
    const { wrapper, command, fetchMock } = await setup();
    command.mockResolvedValue({ ok: false, json: async () => { throw new SyntaxError("private backend HTML"); } } as unknown as Response);
    await click(wrapper, "修改");
    await dialog().get("form").trigger("submit");
    await settle();
    expect(wrapper.emitted("notice")?.flat()).toContain("操作失败，请稍后重试。");
    expect(JSON.stringify(wrapper.emitted("notice"))).not.toContain("private backend");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memories")).toHaveLength(2);
    expect(dialog().exists()).toBe(true);
    wrapper.unmount();
  });
});
