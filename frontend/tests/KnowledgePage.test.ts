import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import KnowledgePage from "../src/KnowledgePage.vue";

function response(payload: unknown, ok = true): Response {
  return { ok, json: async () => payload } as Response;
}

async function settle(): Promise<void> {
  await flushPromises();
  await flushPromises();
}

const knowledgeBase = [{ id: 4, name: "Study", created_at: "x" }];

function indexStatus(
  indexStatusValue: string,
  jobStatus: string | null = null,
  processed = 0,
  total = 2,
) {
  return {
    index_status: indexStatusValue,
    latest_job: jobStatus === null ? null : {
      id: 9,
      status: jobStatus,
      stage: jobStatus === "RUNNING" ? "EMBEDDING" : null,
      processed_chunk_count: processed,
      total_chunk_count: total,
      error_code: jobStatus === "FAILED" ? "INDEX_BUILD_FAILED" : null,
      error_message: jobStatus === "FAILED" || jobStatus === "INTERRUPTED" ? "索引建立失败，请重试。" : null,
      created_at: "x",
      started_at: "x",
      finished_at: jobStatus === "PENDING" || jobStatus === "RUNNING" ? null : "x",
    },
  };
}

describe("KnowledgePage", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("uploads once, then refreshes and selects the document returned by id", async () => {
    const first = { id: 1, name: "Old", created_at: "x", source: { document_version_id: 11, filename: "old.md", media_type: "text/markdown", size_bytes: 1, sha256: "a", created_at: "x" } };
    const created = { id: 2, name: "New", created_at: "x", source: { document_version_id: 12, filename: "new.md", media_type: "text/markdown", size_bytes: 2, sha256: "b", created_at: "x" } };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([first]))
      .mockResolvedValueOnce(response({ index_status: "CHUNKED", latest_job: null }))
      .mockResolvedValueOnce(response(created))
      .mockResolvedValueOnce(response([created, first]));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();

    const input = wrapper.get('input[type="file"]');
    Object.defineProperty(input.element, "files", {
      value: [new File(["# new"], "new.md", { type: "text/markdown" })],
    });
    await input.trigger("change");
    await wrapper.findAll("form")[1]!.trigger("submit");
    await settle();

    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
    expect(wrapper.text()).toContain("New");
    expect(wrapper.text()).toContain("new.md");
    wrapper.unmount();
  });

  it("does not retry an ambiguous upload and tells the user to refresh", async () => {
    const notice = vi.fn();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(response({ index_status: "CHUNKED", latest_job: null }))
      .mockRejectedValueOnce(new TypeError("network"));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage, { attrs: { onNotice: notice } });
    await settle();
    const input = wrapper.get('input[type="file"]');
    Object.defineProperty(input.element, "files", {
      value: [new File(["# new"], "new.md")],
    });
    await input.trigger("change");
    await wrapper.findAll("form")[1]!.trigger("submit");
    await settle();
    expect(notice).toHaveBeenCalledWith("上传结果不明确，请先刷新文档列表确认后再试。");
    expect(fetchMock).toHaveBeenCalledTimes(4);
    wrapper.unmount();
  });

  it("treats an ambiguous admission failure as refresh-first without retrying", async () => {
    const notice = vi.fn();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(response({ index_status: "CHUNKED", latest_job: null }))
      .mockResolvedValueOnce(response({ detail: { code: "KNOWLEDGE_ADMISSION_FAILED" } }, false));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage, { attrs: { onNotice: notice } });
    await settle();
    const input = wrapper.get('input[type="file"]');
    Object.defineProperty(input.element, "files", { value: [new File(["# new"], "new.md")] });
    await input.trigger("change");
    await wrapper.findAll("form")[1]!.trigger("submit");
    await settle();
    expect(notice).toHaveBeenCalledWith("上传结果不明确，请先刷新文档列表确认后再试。");
    expect(fetchMock).toHaveBeenCalledTimes(4);
    wrapper.unmount();
  });

  it("renders live verification ephemerally and resets it after authoritative reload", async () => {
    const document = { id: 7, name: "Note", created_at: "x", source: { document_version_id: 17, filename: "note.md", media_type: "text/markdown", size_bytes: 7, sha256: "abc", created_at: "upload-time" } };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([document]))
      .mockResolvedValueOnce(response({ index_status: "CHUNKED", latest_job: null }))
      .mockResolvedValueOnce(response({ document_version_id: 17, verified: true, verified_at: "verified-time" }))
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([document]))
      .mockResolvedValueOnce(response({ index_status: "CHUNKED", latest_job: null }));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("原文件完整性：尚未验证");
    const verify = wrapper.findAll("button").find((item) => item.text().includes("验证原始文件"));
    await verify!.trigger("click");
    await settle();
    expect(wrapper.text()).toContain("刚刚验证通过");
    expect(wrapper.text()).toContain("verified-time");
    await (wrapper.vm as unknown as { load(): Promise<void> }).load();
    await settle();
    expect(wrapper.text()).toContain("原文件完整性：尚未验证");
    wrapper.unmount();
  });

  it("offers a CHUNKED build, submits one POST, and renders active progress", async () => {
    let statusCall = 0;
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") {
        statusCall += 1;
        return response(statusCall === 1 ? indexStatus("CHUNKED") : indexStatus("INDEXING", "PENDING"));
      }
      if (path === "/api/knowledge-bases/4/index-build" && init?.method === "POST") return response({ job_id: 9 });
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();

    const build = wrapper.findAll("button").find((item) => item.text().includes("建立索引"));
    expect(build?.attributes("disabled")).toBeUndefined();
    await build!.trigger("click");
    await settle();

    expect(fetchMock.mock.calls.filter(([path, init]) => path === "/api/knowledge-bases/4/index-build" && init?.method === "POST")).toHaveLength(1);
    expect(wrapper.text()).toContain("正在建立索引");
    expect(wrapper.text()).toContain("进度：0 / 2");
    wrapper.unmount();
  });

  it("polls an active build to READY, then stops polling and clears the timer on unmount", async () => {
    vi.useFakeTimers();
    let statusCall = 0;
    const clearTimeoutSpy = vi.spyOn(window, "clearTimeout");
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") {
        statusCall += 1;
        return response(statusCall === 1 ? indexStatus("INDEXING", "RUNNING", 1, 2) : indexStatus("READY", "SUCCEEDED", 2, 2));
      }
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("进度：1 / 2");

    await vi.advanceTimersByTimeAsync(3000);
    await settle();
    expect(wrapper.text()).toContain("可检索");
    const settledCalls = statusCall;
    await vi.advanceTimersByTimeAsync(3000);
    expect(statusCall).toBe(settledCalls);

    wrapper.unmount();
    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it("clears an active index polling timer when unmounted", async () => {
    vi.useFakeTimers();
    const clearTimeoutSpy = vi.spyOn(window, "clearTimeout");
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("INDEXING", "PENDING"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    wrapper.unmount();
    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1);
  });

  it.each(["FAILED", "INTERRUPTED"])("stops polling for %s and permits one retry", async (terminalStatus) => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("FAILED", terminalStatus));
      if (path === "/api/knowledge-bases/4/index-build" && init?.method === "POST") return response({ job_id: 10 });
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("索引建立失败，请重试。");
    const build = wrapper.findAll("button").find((item) => item.text().includes("建立索引"));
    expect(build?.attributes("disabled")).toBeUndefined();
    await build!.trigger("click");
    await settle();
    expect(fetchMock.mock.calls.filter(([path, init]) => path === "/api/knowledge-bases/4/index-build" && init?.method === "POST")).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(3000);
    wrapper.unmount();
  });

  it.each(["CHUNKED", "STALE"])("keeps manual build available for %s", async (state) => {
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus(state));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const build = wrapper.findAll("button").find((item) => item.text().includes("建立索引"));
    expect(build?.attributes("disabled")).toBeUndefined();
    if (state === "STALE") expect(wrapper.text()).toContain("需要重建");
    wrapper.unmount();
  });

  it("prevents a duplicate UI submission while the index is active", async () => {
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("INDEXING", "PENDING"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const build = wrapper.findAll("button").find((item) => item.text().includes("建立索引"));
    expect(build?.attributes("disabled")).toBeDefined();
    await build!.trigger("click");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/knowledge-bases/4/index-build")).toHaveLength(0);
    wrapper.unmount();
  });
});
