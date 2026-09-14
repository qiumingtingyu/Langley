import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import WorkspacePanel from "../src/components/WorkspacePanel.vue";

const response = (payload: unknown) => ({ ok: true, json: async () => payload });
afterEach(() => vi.unstubAllGlobals());

describe("Workspace panel", () => {
  it("opens the first conversation after empty creation and sends only relative import paths", async () => {
    const fetch = vi.fn(async (url: string) => response(url === "/api/workspaces" ? [] : {}));
    vi.stubGlobal("fetch", fetch);
    const wrapper = mount(WorkspacePanel, { props: { conversations: [], workspaceId: null, changes: null } });
    await flushPromises();
    fetch.mockImplementation(async (_url: string, init?: RequestInit) => response(init?.method === "POST" ? { conversation_id: 9, skipped: {} } : []));
    await wrapper.get('input[maxlength="255"]').setValue("demo");
    await wrapper.findAll("button").find(b => b.text() === "新建工作区")!.trigger("click");
    await flushPromises();
    expect(wrapper.emitted("open")?.[0]).toEqual([9]);
    const file = new File(["hello"], "hello.txt");
    Object.defineProperty(file, "webkitRelativePath", { value: "picked/nested/hello.txt" });
    const input = wrapper.get('input[type="file"]');
    Object.defineProperty(input.element, "files", { value: [file] });
    await input.trigger("change");
    await flushPromises();
    const call = fetch.mock.calls.find(([url]) => url === "/api/workspaces/import");
    const body = (call?.[1] as RequestInit).body as FormData;
    expect((body.get("files") as File).name).toBe("nested/hello.txt");
    expect(wrapper.text()).toContain("原文件夹不受后续编辑影响");
    wrapper.unmount();
  });

  it("discards stale file results and displays the current run change summary", async () => {
    let resolveOld!: (value: ReturnType<typeof response>) => void;
    const old = new Promise<ReturnType<typeof response>>(resolve => { resolveOld = resolve; });
    vi.stubGlobal("fetch", vi.fn((url: string) => {
      if (url === "/api/workspaces") return Promise.resolve(response([]));
      if (url.startsWith("/api/workspaces/1/")) return old;
      return Promise.resolve(response({ entries: [{ name: "current.txt", type: "file" }] }));
    }));
    const wrapper = mount(WorkspacePanel, { props: { conversations: [], workspaceId: 1, changes: null } });
    await wrapper.setProps({ workspaceId: 2, changes: { added: ["new.txt"], modified: [], deleted: ["old.txt"], complete: true } });
    await flushPromises();
    resolveOld(response({ entries: [{ name: "stale.txt", type: "file" }] }));
    await flushPromises();
    expect(wrapper.text()).toContain("current.txt");
    expect(wrapper.text()).not.toContain("stale.txt");
    expect(wrapper.text()).toContain("+ new.txt");
    expect(wrapper.text()).toContain("− old.txt");
    wrapper.unmount();
  });
});
