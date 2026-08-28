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
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents" && init?.method !== "POST") return response([created, first]);
      if (path.includes("/chunks?")) return response({ document_version_id: 12, successful_chunk_max_chars: null, suggested_chunk_max_chars: 1200, chunk_count: 0, offset: 0, limit: 50, chunks: [] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      if (path === "/api/knowledge-bases/4/documents" && init?.method === "POST") return response(created);
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();

    const input = wrapper.get('input[type="file"]');
    expect(input.attributes("accept")).toBe(".md,text/markdown");
    Object.defineProperty(input.element, "files", {
      value: [new File(["# new"], "new.md", { type: "text/markdown" })],
    });
    await input.trigger("change");
    expect(wrapper.text()).toContain("已选择：new.md");
    expect(wrapper.text()).toContain("当前支持 Markdown 文件。");
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
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) return response({ document_version_id: 17, successful_chunk_max_chars: null, suggested_chunk_max_chars: 1200, chunk_count: 0, offset: 0, limit: 50, chunks: [] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      if (path.endsWith("/verify-source") && init?.method === "POST") return response({ document_version_id: 17, verified: true, verified_at: "verified-time" });
      throw new Error(`unexpected request: ${path}`);
    });
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

  it("refreshes index status after a successful upload so stale readiness is visible", async () => {
    let indexStatusCalls = 0;
    const created = { id: 2, name: "New", created_at: "x", source: { document_version_id: 12, filename: "new.md", media_type: "text/markdown", size_bytes: 2, sha256: "b", created_at: "x" } };
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents" && init?.method !== "POST") return response([created]);
      if (path.includes("/chunks?")) return response({ document_version_id: 12, successful_chunk_max_chars: null, suggested_chunk_max_chars: 1200, chunk_count: 0, offset: 0, limit: 50, chunks: [] });
      if (path === "/api/knowledge-bases/4/index-status") { indexStatusCalls += 1; return response(indexStatus(indexStatusCalls === 1 ? "READY" : "STALE")); }
      if (path === "/api/knowledge-bases/4/documents" && init?.method === "POST") return response(created);
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const input = wrapper.get('input[type="file"]');
    Object.defineProperty(input.element, "files", { value: [new File(["# new"], "new.md")] });
    await input.trigger("change");
    await wrapper.findAll("form")[1]!.trigger("submit");
    await settle();
    expect(indexStatusCalls).toBe(2);
    expect(wrapper.text()).toContain("索引状态：需要重建");
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
    expect(wrapper.text()).toContain("索引状态：正在建立");
    expect(wrapper.text()).toContain("正在建立索引 · 等待执行 · 0 / 2");
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
    expect(wrapper.text()).toContain("正在建立索引 · EMBEDDING · 1 / 2");

    await vi.advanceTimersByTimeAsync(5000);
    await settle();
    expect(wrapper.text()).toContain("可检索");
    const settledCalls = statusCall;
    await vi.advanceTimersByTimeAsync(5000);
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
    await vi.advanceTimersByTimeAsync(5000);
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

  it("keeps current stale readiness distinct from a previously succeeded build", async () => {
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("STALE", "SUCCEEDED", 120, 120));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("索引状态：需要重建");
    expect(wrapper.text()).not.toContain("最近一次构建：已完成");
    expect(wrapper.text()).not.toContain("索引已就绪，可用于后续检索。");
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

  it("shows an actionable error when unprocessed documents block index admission", async () => {
    const notice = vi.fn();
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([]);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      if (path === "/api/knowledge-bases/4/index-build" && init?.method === "POST") return response({ detail: { code: "KNOWLEDGE_BASE_DOCUMENTS_UNPROCESSED" } }, false);
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage, { attrs: { onNotice: notice } });
    await settle();
    await wrapper.findAll("button").find((item) => item.text().includes("建立索引"))!.trigger("click");
    await settle();
    expect(notice).toHaveBeenCalledWith("还有文档尚未处理，请先完成文档处理后再建立索引。");
    wrapper.unmount();
  });
});

describe("KnowledgePage processing bridge", () => {
  afterEach(() => vi.unstubAllGlobals());

  const document = { id: 7, name: "Note", created_at: "x", source: { document_version_id: 17, filename: "note.md", media_type: "text/markdown", size_bytes: 7, sha256: "abc", created_at: "upload-time" } };
  const chunk = { ordinal: 1, content: "A complete chunk content that can be expanded.", heading_path: ["操作系统", "进程"], source_regions: [{ kind: "text_span", start_byte: 12, end_byte: 48 }] };
  const chunksPage = (successful: number | null, chunkCount = 1, offset = 0) => ({ document_version_id: 17, successful_chunk_max_chars: successful, suggested_chunk_max_chars: 1200, chunk_count: chunkCount, offset, limit: 50, chunks: chunkCount === 0 ? [] : [chunk] });

  function defaultFetch(page = chunksPage(null), state = "CHUNKED") {
    return vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) return response(page);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus(state));
      throw new Error(`unexpected request: ${path}`);
    });
  }

  it("initializes the backend suggestion, distinguishes processed empty chunks, and keeps the current config separate from the draft", async () => {
    const fetchMock = defaultFetch(chunksPage(1200, 0));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("当前分块配置：1200");
    expect(wrapper.text()).toContain("重新切片");
    const processingInput = wrapper.findAll("input").find((item) => item.attributes("inputmode") === "numeric")!;
    await processingInput.setValue("800");
    expect(wrapper.text()).toContain("当前分块配置：1200");
    expect(wrapper.text()).toContain("单个分块最大字符数");
    wrapper.unmount();
  });

  it("shows unknown historical configuration truthfully and renders chunk facts with expand", async () => {
    const fetchMock = defaultFetch(chunksPage(null, 1));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("当前切片配置未知，请重新处理文档。");
    expect(wrapper.text()).toContain("处理文档");
    expect((wrapper.findAll("input").find((item) => item.attributes("inputmode") === "numeric")!.element as HTMLInputElement).value).toBe("1200");
    expect(wrapper.text()).toContain("#1");
    expect(wrapper.text()).toContain("操作系统 › 进程");
    expect(wrapper.text()).toContain("text_span [12, 48)");
    await wrapper.findAll("button").find((item) => item.text() === "展开")!.trigger("click");
    expect(wrapper.text()).toContain(chunk.content);
    wrapper.unmount();
  });

  it("keeps old chunks visible while pending and applies only a successful rebuild before reloading page zero", async () => {
    let resolveRebuild: ((value: Response) => void) | undefined;
    const rebuild = new Promise<Response>((resolve) => { resolveRebuild = resolve; });
    let chunkReads = 0;
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) { chunkReads += 1; return response(chunkReads === 1 ? chunksPage(1200) : chunksPage(800)); }
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus(chunkReads > 1 ? "STALE" : "CHUNKED"));
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return rebuild;
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const input = wrapper.findAll("input").find((item) => item.attributes("inputmode") === "numeric")!;
    await input.setValue("800");
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    expect(input.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("#1");
    resolveRebuild!(response({ document_version_id: 17, successful_chunk_max_chars: 800, chunk_count: 1, resulting_index_status: "STALE" }));
    await settle();
    expect(wrapper.text()).toContain("当前分块配置：800");
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/document-versions/17/chunks?offset=0&limit=50")).toBe(true);
    expect(wrapper.text()).toContain("需要重建");
    wrapper.unmount();
  });

  it("retains durable facts on ordinary failure and fast-disables while INDEXING", async () => {
    const fetchMock = defaultFetch(chunksPage(1200), "INDEXING");
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    expect(wrapper.text()).toContain("正在建立索引，请等待完成后再重新处理文档。");
    const button = wrapper.findAll("button").find((item) => item.text().includes("重新切片"))!;
    expect(button.attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });

  it("handles an INDEXING race without fake success and refreshes status", async () => {
    let statusCalls = 0;
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) return response(chunksPage(1200));
      if (path === "/api/knowledge-bases/4/index-status") { statusCalls += 1; return response(indexStatus(statusCalls === 1 ? "CHUNKED" : "INDEXING")); }
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return response({ detail: { code: "KNOWLEDGE_BASE_INDEXING" } }, false);
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await settle();
    expect(wrapper.text()).toContain("知识库正在建立索引，本次重新切片未生效。");
    expect(wrapper.text()).toContain("当前分块配置：1200");
    expect(statusCalls).toBe(2);
    wrapper.unmount();
  });

  it("retains old durable chunks and the user draft after an ordinary rebuild failure", async () => {
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) return response(chunksPage(1200));
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return response({ detail: { code: "VALIDATION_ERROR" } }, false);
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const input = wrapper.findAll("input").find((item) => item.attributes("inputmode") === "numeric")!;
    await input.setValue("800");
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await settle();
    expect(wrapper.text()).toContain("输入内容无效，请检查后重试。");
    expect(wrapper.text()).toContain("当前分块配置：1200");
    expect(wrapper.text()).toContain("#1");
    expect((input.element as HTMLInputElement).value).toBe("800");
    wrapper.unmount();
  });

  it("paginates within the bounded range", async () => {
    const second = { ...chunk, ordinal: 51 };
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.endsWith("offset=0&limit=50")) return response({ ...chunksPage(1200, 51), chunks: [chunk] });
      if (path.endsWith("offset=50&limit=50")) return response({ ...chunksPage(1200, 51, 50), chunks: [second] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const next = wrapper.findAll("button").find((item) => item.text() === "下一页")!;
    expect(wrapper.findAll("button").filter((item) => item.text() === "下一页")).toHaveLength(2);
    expect(next.attributes("disabled")).toBeUndefined();
    await next.trigger("click");
    await settle();
    expect(wrapper.text()).toContain("显示 51–51");
    expect(wrapper.text()).toContain("#51");
    expect(next.attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });

  it("does not overwrite a local processing draft while paginating", async () => {
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.endsWith("offset=0&limit=50")) return response({ ...chunksPage(1200, 51), chunks: [chunk] });
      if (path.endsWith("offset=50&limit=50")) return response({ ...chunksPage(1200, 51, 50), chunks: [{ ...chunk, ordinal: 51 }] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    const input = wrapper.findAll("input").find((item) => item.attributes("inputmode") === "numeric")!;
    await input.setValue("800");
    await wrapper.findAll("button").find((item) => item.text() === "下一页")!.trigger("click");
    await settle();
    expect((input.element as HTMLInputElement).value).toBe("800");
    expect(wrapper.text()).toContain("当前分块配置：1200");
    wrapper.unmount();
  });

  it("keeps a confirmed rebuild success when its follow-up chunk reload fails", async () => {
    let chunkReads = 0;
    let statusCalls = 0;
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document]);
      if (path.includes("/chunks?")) { chunkReads += 1; if (chunkReads === 1) return response(chunksPage(1200)); throw new TypeError("reload failed"); }
      if (path === "/api/knowledge-bases/4/index-status") { statusCalls += 1; return response(indexStatus(statusCalls === 1 ? "CHUNKED" : "STALE")); }
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return response({ document_version_id: 17, successful_chunk_max_chars: 800, chunk_count: 2, resulting_index_status: "STALE" });
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await settle();
    expect(wrapper.text()).toContain("当前分块配置：800");
    expect(wrapper.text()).toContain("处理已完成，但暂时无法刷新 Chunk，请刷新后查看。");
    expect(wrapper.text()).toContain("需要重建");
    expect(wrapper.text()).not.toContain("尚未处理");
    expect(wrapper.text()).not.toContain("#1");
    wrapper.unmount();
  });

  it("ignores a late chunk page from an old document selection", async () => {
    const documentB = { ...document, id: 8, name: "New", source: { ...document.source, document_version_id: 18, filename: "new.md" } };
    let resolveA: ((value: Response) => void) | undefined;
    const pendingA = new Promise<Response>((resolve) => { resolveA = resolve; });
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document, documentB]);
      if (path.includes("/17/chunks?")) return pendingA;
      if (path.includes("/18/chunks?")) return response({ ...chunksPage(800), document_version_id: 18, chunks: [{ ...chunk, ordinal: 8, content: "B chunk" }] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await flushPromises();
    await wrapper.findAll("button").find((item) => item.text().includes("New"))!.trigger("click");
    resolveA!(response(chunksPage(1200)));
    await settle();
    expect(wrapper.text()).toContain("当前分块配置：800");
    expect(wrapper.text()).toContain("B chunk");
    expect(wrapper.text()).not.toContain("当前分块配置：1200");
    wrapper.unmount();
  });

  it("does not let a stale rebuild result overwrite the newly selected document", async () => {
    const documentB = { ...document, id: 8, name: "New", source: { ...document.source, document_version_id: 18, filename: "new.md" } };
    let resolveRebuild: ((value: Response) => void) | undefined;
    const pendingRebuild = new Promise<Response>((resolve) => { resolveRebuild = resolve; });
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document, documentB]);
      if (path.includes("/17/chunks?")) return response(chunksPage(1200));
      if (path.includes("/18/chunks?")) return response({ ...chunksPage(800), document_version_id: 18, chunks: [{ ...chunk, ordinal: 8, content: "B chunk" }] });
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return pendingRebuild;
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await wrapper.findAll("button").find((item) => item.text().includes("New"))!.trigger("click");
    await settle();
    resolveRebuild!(response({ document_version_id: 17, successful_chunk_max_chars: 600, chunk_count: 1, resulting_index_status: "STALE" }));
    await settle();
    expect(wrapper.text()).toContain("当前分块配置：800");
    expect(wrapper.text()).toContain("B chunk");
    expect(wrapper.findAll("button").find((item) => item.text().includes("重新切片"))?.attributes("disabled")).toBeUndefined();
    wrapper.unmount();
  });

  it("refreshes same-KB index status after a stale document rebuild result", async () => {
    const documentB = { ...document, id: 8, name: "New", source: { ...document.source, document_version_id: 18, filename: "new.md" } };
    let resolveRebuild: ((value: Response) => void) | undefined;
    const pendingRebuild = new Promise<Response>((resolve) => { resolveRebuild = resolve; });
    let statusCalls = 0;
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") return response([document, documentB]);
      if (path.includes("/17/chunks?")) return response(chunksPage(1200));
      if (path.includes("/18/chunks?")) return response({ ...chunksPage(800), document_version_id: 18, chunks: [{ ...chunk, ordinal: 8, content: "B chunk" }] });
      if (path === "/api/knowledge-bases/4/index-status") { statusCalls += 1; return response(indexStatus(statusCalls === 1 ? "READY" : "STALE")); }
      if (path.endsWith("/chunks/rebuild") && init?.method === "POST") return pendingRebuild;
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage);
    await settle();
    await wrapper.findAll("form").at(-1)!.trigger("submit");
    await wrapper.findAll("button").find((item) => item.text().includes("New"))!.trigger("click");
    await settle();
    resolveRebuild!(response({ document_version_id: 17, successful_chunk_max_chars: 600, chunk_count: 1, resulting_index_status: "STALE" }));
    await settle();
    expect(wrapper.text()).toContain("B chunk");
    expect(wrapper.text()).toContain("当前分块配置：800");
    expect(wrapper.text()).toContain("需要重建");
    expect(statusCalls).toBe(2);
    wrapper.unmount();
  });

  it("deselects the current document after its chunk endpoint returns 404", async () => {
    const notice = vi.fn();
    let documentsCalls = 0;
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/knowledge-bases") return response(knowledgeBase);
      if (path === "/api/knowledge-bases/4/documents") { documentsCalls += 1; return response(documentsCalls === 1 ? [document] : []); }
      if (path.includes("/chunks?")) return response({ detail: { code: "DOCUMENT_VERSION_NOT_FOUND" } }, false);
      if (path === "/api/knowledge-bases/4/index-status") return response(indexStatus("CHUNKED"));
      throw new Error(`unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(KnowledgePage, { attrs: { onNotice: notice } });
    await settle();
    expect(wrapper.text()).not.toContain("文档处理");
    expect(notice).toHaveBeenCalledWith("该知识库或文档已不存在，请刷新后重试。");
    wrapper.unmount();
  });
});
