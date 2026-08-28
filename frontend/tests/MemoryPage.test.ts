import { flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import MemoryPage from "../src/MemoryPage.vue";

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

function memory(id: number, content: string) {
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
    vi.stubGlobal("confirm", () => true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
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
    expect(wrapper.text()).toContain("状态：自动整理可用");
    expect(wrapper.text()).toContain("待整理内容：2 条");
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

    expect(wrapper.text()).toContain("待整理内容：1 条");
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
    expect(wrapper.text()).toContain("待整理内容：3 条");
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
    await wrapper.get('textarea[maxlength="1000"]').setValue("新的长期信息");
    await wrapper.find("form").trigger("submit");
    await settle();

    expect(wrapper.text()).toContain("新的长期信息");
    expect(wrapper.text()).not.toContain("命令响应不应成为列表权威");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/memory-status")).toHaveLength(2);
    wrapper.unmount();
  });
});
