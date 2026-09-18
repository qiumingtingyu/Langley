import { flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

import SkillsPage from "../src/SkillsPage.vue";

function response(payload: unknown, ok = true): Response {
  return { ok, json: async () => payload } as Response;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

function summary(name: string) {
  return { name, description: `${name} description`, source: "BUILTIN" as const };
}

function detail(name: string, instructions = "# Procedure") {
  return { ...summary(name), instructions, resources: [] };
}

function button(wrapper: ReturnType<typeof mount>, label: string) {
  return wrapper.findAll("button").find((item) => item.text().includes(label))!;
}

describe("SkillsPage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads the builtin inventory, selects the first Skill, and renders its detail", async () => {
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/skills") return Promise.resolve(response([summary("interview-review"), summary("study-plan")]));
      if (path === "/api/skills/interview-review") {
        return Promise.resolve(response({
          ...detail("interview-review", "# Review\n\n- Read evidence\n- Apply rubric"),
          resources: [{ path: "references/rubric.md", byte_size: 4301 }],
        }));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(SkillsPage);
    expect(wrapper.text()).toContain("正在读取技能目录");
    await settle();

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/skills",
      "/api/skills/interview-review",
    ]);
    expect(wrapper.text()).toContain("interview-review");
    expect(wrapper.text()).toContain("内置");
    expect(wrapper.get("h1").text()).toBe("技能");
    expect(wrapper.get(".skill-instructions h1").text()).toBe("Review");
    expect(wrapper.text()).toContain("references/rubric.md");
    expect(wrapper.text()).toContain("4.2 KiB");
    expect(button(wrapper, "interview-review").attributes("aria-pressed")).toBe("true");
  });

  it("renders safe Markdown links, escapes raw HTML, and shows empty references", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) => Promise.resolve(response(
      path === "/api/skills"
        ? [summary("safe-doc")]
        : detail(
            "safe-doc",
            "<script>window.pwned = true</script>\n\n[外部](https://example.com/docs) [内部](references/local.md) ![图](https://example.com/a.png)",
          ),
    ))));

    const wrapper = mount(SkillsPage);
    await settle();

    expect(wrapper.find("script").exists()).toBe(false);
    expect(wrapper.html()).toContain("&lt;script&gt;");
    const link = wrapper.get('.skill-instructions a[href="https://example.com/docs"]');
    expect(link.attributes("target")).toBe("_blank");
    expect(link.attributes("rel")).toBe("noopener noreferrer");
    expect(wrapper.find('.skill-instructions a[href="references/local.md"]').exists()).toBe(false);
    expect(wrapper.find(".skill-instructions img").exists()).toBe(false);
    expect(wrapper.text()).toContain("当前 Skill 没有附带 references。");
  });

  it("maps catalog failure safely and retries", async () => {
    let attempt = 0;
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/skills" && attempt++ === 0) {
        return Promise.resolve(response({ detail: { code: "SKILL_CATALOG_UNAVAILABLE", private: "do not render" } }, false));
      }
      if (path === "/api/skills") return Promise.resolve(response([summary("recovered")]));
      return Promise.resolve(response(detail("recovered")));
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(SkillsPage);
    await settle();
    expect(wrapper.get('[role="alert"]').text()).toContain("技能目录暂时不可用，请稍后重试。");
    expect(wrapper.text()).not.toContain("do not render");

    await button(wrapper, "重试").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("recovered");
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/skills")).toHaveLength(2);
  });

  it("maps detail failure and retries the selected Skill", async () => {
    let detailAttempt = 0;
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/skills") return Promise.resolve(response([summary("fragile")]));
      if (detailAttempt++ === 0) {
        return Promise.resolve(response({ detail: { code: "SKILL_CONTENT_UNAVAILABLE" } }, false));
      }
      return Promise.resolve(response(detail("fragile", "## Restored")));
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(SkillsPage);
    await settle();
    expect(wrapper.get('[role="alert"]').text()).toContain("技能内容暂时不可用，请稍后重试。");

    await button(wrapper, "重试内容").trigger("click");
    await settle();
    expect(wrapper.get(".skill-instructions h2").text()).toBe("Restored");
  });

  it("refreshes inventory after a missing Skill and selects the available default", async () => {
    let inventoryAttempt = 0;
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/skills") {
        inventoryAttempt += 1;
        return Promise.resolve(response(inventoryAttempt === 1
          ? [summary("removed"), summary("available")]
          : [summary("available")]));
      }
      if (path === "/api/skills/removed") {
        return Promise.resolve(response({ detail: { code: "SKILL_NOT_FOUND" } }, false));
      }
      if (path === "/api/skills/available") {
        return Promise.resolve(response(detail("available", "# Available")));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(SkillsPage);
    await settle();
    expect(wrapper.get('[role="alert"]').text()).toContain("这个内置技能已不存在，请刷新后重试。");

    await button(wrapper, "刷新技能目录").trigger("click");
    await settle();

    expect(wrapper.get(".skill-instructions h1").text()).toBe("Available");
    expect(button(wrapper, "available").attributes("aria-pressed")).toBe("true");
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/skills",
      "/api/skills/removed",
      "/api/skills",
      "/api/skills/available",
    ]);
  });

  it("does not let an older detail response replace the current selection", async () => {
    const oldDetail = deferred<Response>();
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/skills") return Promise.resolve(response([summary("alpha"), summary("beta")]));
      if (path === "/api/skills/alpha") return oldDetail.promise;
      if (path === "/api/skills/beta") return Promise.resolve(response(detail("beta", "# Current beta")));
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const wrapper = mount(SkillsPage);
    await flushPromises();
    await nextTick();
    await button(wrapper, "beta").trigger("click");
    await settle();
    oldDetail.resolve(response(detail("alpha", "# Stale alpha")));
    await settle();

    expect(wrapper.text()).toContain("Current beta");
    expect(wrapper.text()).not.toContain("Stale alpha");
    expect(button(wrapper, "beta").attributes("aria-pressed")).toBe("true");
  });

  it("shows an explicit empty inventory state", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(response([]))));
    const wrapper = mount(SkillsPage);
    await settle();
    expect(wrapper.text()).toContain("当前没有内置 Skill。");
  });
});
