import { DOMWrapper, flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import WorkspacePanel from "../src/components/WorkspacePanel.vue";
import type { Conversation } from "../src/types";

const response = (payload: unknown) => ({ ok: true, json: async () => payload }) as Response;
const workspaceList = [{ id: 1, name: "Project A" }, { id: 2, name: "Project B" }];
const conversation = (id: number, workspaceId: number): Conversation => ({
  id, workspace_id: workspaceId, title: "Project chat", created_at: "x", updated_at: "x", last_message_at: null,
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
const mounted: VueWrapper[] = [];
function setup(workspaceId: number | null = 1, conversations: Conversation[] = []) {
  const wrapper = mount(WorkspacePanel, {
    props: { open: true, conversationId: 10, conversations, workspaceId, changes: null },
    attachTo: document.body,
  });
  mounted.push(wrapper);
  return wrapper;
}
function panel() {
  return new DOMWrapper(document.querySelector<HTMLElement>('[role="dialog"]')!);
}
async function click(label: string) {
  const button = panel().findAll("button").find(b => b.text() === label);
  expect(button).toBeDefined();
  await button!.trigger("click");
  await flushPromises();
}
function mockFiles(handler?: (url: URL) => Response | Promise<Response>) {
  const fetchMock = vi.fn(async (path: string) => {
    if (path === "/api/workspaces") return response(workspaceList);
    return handler?.(new URL(path, "http://localhost")) ?? response({ entries: [] });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
afterEach(() => {
  mounted.splice(0).forEach(wrapper => wrapper.unmount());
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

describe("Workspace panel", () => {
  it("clears a workspace-list alert after a successful explicit refresh", async () => {
    let listReads = 0;
    const fetchMock = vi.fn(async () => {
      listReads++;
      if (listReads === 1) throw new TypeError("offline");
      return response(workspaceList);
    });
    vi.stubGlobal("fetch", fetchMock);
    setup(null);
    await flushPromises();
    expect(panel().get('[role="alert"]').text()).toContain("工作区列表暂时不可用");
    await click("刷新工作区列表");
    expect(panel().find('[role="alert"]').exists()).toBe(false);
    expect(panel().get("select").text()).toContain("Project A");
    expect(fetchMock.mock.calls).toHaveLength(2);
  });

  it("shows unsupported binary preview locally once and clears it when another path is selected", async () => {
    const nextText = deferred<Response>();
    let textReads = 0;
    const fetchMock = mockFiles(url => {
      if (url.searchParams.get("preview") === "false") return response({ entries: [
        { name: "readme.txt", type: "file" }, { name: "photo.png", type: "file" },
      ] });
      if (url.searchParams.get("path") === "photo.png") return {
        ok: false, json: async () => ({ detail: { code: "TEXT_FILE_UNSUPPORTED" } }),
      } as Response;
      textReads++;
      return textReads === 1 ? response({ path: "readme.txt", content: "Old text", end_line: 1, total_lines: 1 }) : nextText.promise;
    });
    setup();
    await flushPromises();
    await click("readme.txt");
    expect(panel().get("pre").text()).toBe("Old text");
    const beforeBinary = fetchMock.mock.calls.length;
    await click("photo.png");
    expect(panel().get('[aria-label="文件预览"]').text()).toContain("此文件不支持文本预览。二进制文件仍保留在工作区中。");
    expect(panel().find('[role="alert"]').exists()).toBe(false);
    expect(panel().find("pre").exists()).toBe(false);
    expect(panel().get('button[aria-current="true"]').text()).toBe("photo.png");
    await flushPromises();
    expect(fetchMock.mock.calls).toHaveLength(beforeBinary + 1);
    await click("readme.txt");
    expect(panel().text()).not.toContain("此文件不支持文本预览");
    nextText.resolve(response({ path: "readme.txt", content: "New text", end_line: 1, total_lines: 1 }));
    await flushPromises();
    expect(panel().get("pre").text()).toBe("New text");
  });

  it("clears binary preview feedback on workspace switching and discards old unsupported responses", async () => {
    const delayed = deferred<Response>();
    let previews = 0;
    const unsupported = { ok: false, json: async () => ({ detail: { code: "TEXT_FILE_UNSUPPORTED" } }) } as Response;
    mockFiles(url => {
      if (url.searchParams.get("preview") === "true") return ++previews === 1 ? unsupported : delayed.promise;
      return response({ entries: [{ name: "archive.zip", type: "file" }] });
    });
    const wrapper = setup();
    await flushPromises();
    await click("archive.zip");
    expect(panel().text()).toContain("此文件不支持文本预览");
    await wrapper.setProps({ workspaceId: 2, conversationId: 20 });
    await flushPromises();
    expect(panel().text()).not.toContain("此文件不支持文本预览");
    await click("archive.zip");
    await wrapper.setProps({ workspaceId: 1, conversationId: 10 });
    await flushPromises();
    delayed.resolve(unsupported);
    await flushPromises();
    expect(panel().find('[aria-label="文件预览"]').exists()).toBe(false);
    expect(panel().find('[role="alert"]').exists()).toBe(false);
  });

  it("preserves a real file error when the workspace list refresh succeeds", async () => {
    mockFiles(url => url.searchParams.get("preview") === "true" ? {
      ok: false, json: async () => ({ detail: { code: "WORKSPACE_NOT_FOUND" } }),
    } as Response : response({ entries: [{ name: "gone.txt", type: "file" }] }));
    setup();
    await flushPromises();
    await click("gone.txt");
    expect(panel().get('[role="alert"]').text()).toContain("文件暂时无法读取");
    await click("刷新工作区列表");
    expect(panel().get('[role="alert"]').text()).toContain("文件暂时无法读取");
    expect(panel().text()).not.toContain("二进制文件仍保留在工作区中");
  });

  it("does not restore a stale workspace-list failure after a newer successful read", async () => {
    const oldRead = deferred<Response>();
    let reads = 0;
    vi.stubGlobal("fetch", vi.fn(async () => ++reads === 1 ? oldRead.promise : response(workspaceList)));
    const wrapper = setup(null);
    await flushPromises();
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    oldRead.resolve({ ok: false, json: async () => ({ detail: { code: "UNAVAILABLE" } }) } as Response);
    await flushPromises();
    expect(panel().find('[role="alert"]').exists()).toBe(false);
    expect(panel().get("select").text()).toContain("Project A");
  });

  it("creates once and imports only relative paths with the managed-copy notice", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) =>
      response(init?.method === "POST" ? { conversation_id: 9, skipped: { cache: 2, sensitive: 1 } } : []));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = setup(null);
    await flushPromises();
    const management = panel().findAll("details").find(d => d.text().includes("工作区管理"))!;
    expect(management.attributes("open")).toBeUndefined();
    await management.get("summary").trigger("click");
    await panel().get('input[maxlength="255"]').setValue("demo");
    await click("新建工作区");
    expect(wrapper.emitted("open")?.[0]).toEqual([9]);
    const file = new File(["hello"], "hello.txt");
    Object.defineProperty(file, "webkitRelativePath", { value: "picked/nested/hello.txt" });
    const input = panel().get('input[type="file"]');
    Object.defineProperty(input.element, "files", { value: [file] });
    await input.trigger("change");
    await flushPromises();
    const body = fetchMock.mock.calls.find(([url]) => url === "/api/workspaces/import")?.[1]?.body as FormData;
    expect((body.get("files") as File).name).toBe("nested/hello.txt");
    expect(panel().text()).toContain("已复制导入，跳过 3 项");
    expect(panel().text()).toContain("原文件夹不受后续编辑影响");
    expect(panel().text()).toContain("导入后使用托管副本，原文件夹不会被后续修改。");
  });

  it("keeps the durable selector value and opens an existing workspace conversation without PATCH", async () => {
    const fetchMock = mockFiles();
    const wrapper = setup(1, [conversation(11, 1), conversation(22, 2)]);
    await flushPromises();
    expect((panel().get("select").element as HTMLSelectElement).value).toBe("1");
    await panel().get("select").setValue("2");
    await flushPromises();
    expect(wrapper.emitted("open")).toEqual([[22]]);
    expect((panel().get("select").element as HTMLSelectElement).value).toBe("1");
    expect(fetchMock.mock.calls.every(([path]) => !path.startsWith("/api/conversations"))).toBe(true);
    await wrapper.setProps({ workspaceId: 2, conversationId: 22 });
    await flushPromises();
    expect((panel().get("select").element as HTMLSelectElement).value).toBe("2");
    expect(panel().text()).toContain("当前会话正在此工作区中运行。");
  });

  it("creates a workspace conversation when none exists and can explicitly start another", async () => {
    const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === "POST") return response({ id: 23 });
      return response(path === "/api/workspaces" ? workspaceList : { entries: [] });
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = setup(1);
    await flushPromises();
    await panel().get("select").setValue("2");
    await flushPromises();
    expect(wrapper.emitted("open")).toEqual([[23]]);
    const calls = fetchMock.mock.calls.filter(([, init]) => init?.method);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual(["/api/conversations", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workspace_id: 2 }),
    }]);
    await click("在此工作区新建会话");
    expect(JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body))).toEqual({ workspace_id: 1 });
  });

  it("discards stale decoded file responses after workspace A to B", async () => {
    const oldBody = deferred<unknown>();
    mockFiles(url => url.pathname.includes("/1/") ?
      { ok: true, json: () => oldBody.promise } as Response :
      response({ entries: [{ name: "current.txt", type: "file" }] }));
    const wrapper = setup();
    await flushPromises();
    await wrapper.setProps({ workspaceId: 2, conversationId: 20 });
    await flushPromises();
    oldBody.resolve({ entries: [{ name: "stale.txt", type: "file" }] });
    await flushPromises();
    expect(panel().text()).toContain("current.txt");
    expect(panel().text()).not.toContain("stale.txt");
  });

  it.each([false, true])("discards old preview when conversation switches (same workspace: %s)", async (sameWorkspace) => {
    const old = deferred<Response>();
    mockFiles(url => url.searchParams.get("preview") === "true" ? old.promise :
      response({ entries: [{ name: "readme.txt", type: "file" }] }));
    const wrapper = setup();
    await flushPromises();
    await click("readme.txt");
    await wrapper.setProps({ conversationId: 20, workspaceId: sameWorkspace ? 1 : 2 });
    await flushPromises();
    old.resolve(response({ path: "readme.txt", content: "old preview", end_line: 1, total_lines: 1 }));
    await flushPromises();
    expect(panel().text()).not.toContain("old preview");
    expect(panel().find('[aria-label="文件预览"]').exists()).toBe(false);
  });

  it("browses directories, blocks symlinks, stays within root, and paginates the preview", async () => {
    const fetchMock = mockFiles(url => {
      const path = url.searchParams.get("path");
      if (url.searchParams.get("preview") === "true") {
        const next = url.searchParams.get("offset") === "3";
        return response({ path, content: next ? "third line" : "first\n  second", end_line: next ? 3 : 2, total_lines: 3 });
      }
      return response({ entries: path === "src" ?
        [{ name: "readme.txt", type: "file" }] :
        [{ name: "src", type: "directory" }, { name: "link", type: "symlink" }] });
    });
    setup();
    await flushPromises();
    expect(panel().findAll("button").find(b => b.text() === "上一级")!.attributes("disabled")).toBeDefined();
    const link = panel().findAll("button").find(b => b.text().includes("link"))!;
    expect(link.attributes("disabled")).toBeDefined();
    const calls = fetchMock.mock.calls.length;
    await link.trigger("click");
    expect(fetchMock.mock.calls).toHaveLength(calls);
    await click("src");
    expect(panel().text()).toContain("/src");
    await click("readme.txt");
    expect(panel().get("pre").text()).toBe("first\n  second");
    expect(panel().get('button[aria-current="true"]').text()).toContain("readme.txt");
    await click("下一页");
    expect(panel().get("pre").text()).toBe("third line");
    expect(panel().text()).not.toContain("下一页");
    await click("上一级");
    expect(panel().find('[aria-label="文件预览"]').exists()).toBe(false);
    expect(panel().findAll("button").find(b => b.text() === "上一级")!.attributes("disabled")).toBeDefined();
    expect(fetchMock.mock.calls.filter(([url]) => url.includes("/files?")).every(([url]) =>
      !new URL(url, "http://localhost").searchParams.get("path")?.includes(".."))).toBe(true);
  });

  it("shows counts, all change kinds, incomplete scanning, and the no-rollback explanation", async () => {
    mockFiles();
    const wrapper = setup();
    await wrapper.setProps({ changes: { added: ["new.py"], modified: ["app.py"], deleted: ["old.py"], complete: false } });
    await flushPromises();
    const changes = panel().get('[aria-label="本次文件变化"]');
    expect(changes.text()).toContain("新增 1 · 修改 1 · 删除 1");
    expect(changes.text()).toContain("新增new.py");
    expect(changes.text()).toContain("修改app.py");
    expect(changes.text()).toContain("删除old.py");
    expect(changes.text()).toContain("扫描达到上限，结果可能不完整。");
    expect(panel().text()).toContain("失败或停止不会撤销文件变化");
    expect(panel().text()).toContain("服务重启后不可恢复");
    expect(panel().text()).not.toContain("本次运行未检测到文件变化。");
  });

  it("distinguishes complete empty changes from absent or incomplete summaries", async () => {
    mockFiles();
    const wrapper = setup();
    await flushPromises();
    expect(panel().find('[aria-label="本次文件变化"]').exists()).toBe(false);
    await wrapper.setProps({ changes: { added: [], modified: [], deleted: [], complete: true } });
    await flushPromises();
    expect(panel().text()).toContain("本次运行未检测到文件变化。");
    await wrapper.setProps({ changes: { added: [], modified: [], deleted: [], complete: false } });
    await flushPromises();
    expect(panel().text()).not.toContain("本次运行未检测到文件变化。");
    expect(panel().text()).toContain("扫描达到上限");
  });

  it("keeps create single-flight across close/reopen without repeating unknown side effects", async () => {
    const pending = deferred<Response>();
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) =>
      init?.method === "POST" ? pending.promise : response(workspaceList));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = setup(null);
    await flushPromises();
    await click("新建工作区");
    await wrapper.setProps({ open: false });
    await flushPromises();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    await wrapper.setProps({ open: true });
    await flushPromises();
    await click("新建工作区");
    await panel().get("select").setValue("2");
    await flushPromises();
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
    pending.resolve({ ok: false, json: async () => { throw new Error("private HTML"); } } as unknown as Response);
    await flushPromises();
    expect(panel().get('[role="alert"]').text()).toContain("未能确认操作结果");
    expect(panel().text()).not.toContain("private HTML");
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it.each(["create", "open"] as const)("ignores stale %s completion after changing conversations", async (operation) => {
    const pending = deferred<Response>();
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) =>
      init?.method === "POST" ? pending.promise : response(url === "/api/workspaces" ? workspaceList : { entries: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = setup(1);
    await flushPromises();
    if (operation === "create") await click("新建工作区");
    else await panel().get("select").setValue("2");
    await wrapper.setProps({ conversationId: 20 });
    await flushPromises();
    pending.resolve(response({ id: 99, conversation_id: 99, skipped: {} }));
    await flushPromises();
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("does not let a file refresh discard a successful workspace creation", async () => {
    const pending = deferred<Response>();
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) =>
      init?.method === "POST" ? pending.promise : response(url === "/api/workspaces" ? workspaceList : { entries: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = setup();
    await flushPromises();
    await click("新建工作区");
    await click("刷新文件");
    pending.resolve(response({ conversation_id: 99, skipped: {} }));
    await flushPromises();
    expect(wrapper.emitted("open")).toEqual([[99]]);
  });

  it("blocks workspace mutations and navigation while App is reading the selected conversation", async () => {
    const fetchMock = mockFiles();
    const wrapper = setup(1, [conversation(22, 2)]);
    await flushPromises();
    await wrapper.setProps({ navigationBusy: true });
    await click("新建工作区");
    await click("在此工作区新建会话");
    await panel().get("select").setValue("2");
    const file = new File(["hello"], "hello.txt");
    Object.defineProperty(file, "webkitRelativePath", { value: "picked/hello.txt" });
    const input = panel().get('input[type="file"]');
    Object.defineProperty(input.element, "files", { value: [file] });
    await input.trigger("change");
    await flushPromises();
    expect(wrapper.emitted("open")).toBeUndefined();
    expect(fetchMock.mock.calls.every(([path]) => path.startsWith("/api/workspaces"))).toBe(true);
    expect(fetchMock.mock.calls.some(([path]) => path.endsWith("/import"))).toBe(false);
  });
});
