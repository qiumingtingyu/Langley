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

describe("KnowledgePage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uploads once, then refreshes and selects the document returned by id", async () => {
    const first = { id: 1, name: "Old", created_at: "x", source: { document_version_id: 11, filename: "old.md", media_type: "text/markdown", size_bytes: 1, sha256: "a", created_at: "x" } };
    const created = { id: 2, name: "New", created_at: "x", source: { document_version_id: 12, filename: "new.md", media_type: "text/markdown", size_bytes: 2, sha256: "b", created_at: "x" } };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([first]))
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
    expect(fetchMock).toHaveBeenCalledTimes(3);
    wrapper.unmount();
  });

  it("treats an ambiguous admission failure as refresh-first without retrying", async () => {
    const notice = vi.fn();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([]))
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
    expect(fetchMock).toHaveBeenCalledTimes(3);
    wrapper.unmount();
  });

  it("renders live verification ephemerally and resets it after authoritative reload", async () => {
    const document = { id: 7, name: "Note", created_at: "x", source: { document_version_id: 17, filename: "note.md", media_type: "text/markdown", size_bytes: 7, sha256: "abc", created_at: "upload-time" } };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([document]))
      .mockResolvedValueOnce(response({ document_version_id: 17, verified: true, verified_at: "verified-time" }))
      .mockResolvedValueOnce(response([{ id: 4, name: "Study", created_at: "x" }]))
      .mockResolvedValueOnce(response([document]));
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
});
