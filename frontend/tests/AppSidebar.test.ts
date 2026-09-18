import { DOMWrapper, flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it } from "vitest";

import AppSidebar from "../src/components/AppSidebar.vue";

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("AppSidebar", () => {
  it.each([false, true])("groups workspace and ordinary conversations without duplicates (mobile: %s)", async (mobile) => {
    const base = { created_at: "x", updated_at: "x", last_message_at: null };
    const wrapper = mount(AppSidebar, {
      props: {
        conversations: [
          { ...base, id: 1, title: "普通 A", workspace_id: null },
          { ...base, id: 2, title: "项目 A", workspace_id: 7 },
          { ...base, id: 3, title: "普通 B" },
          { ...base, id: 4, title: "项目 B", workspace_id: 8 },
        ],
        selectedConversationId: 2, activeView: "chat", memoryUpdated: false, busy: false, loading: false,
      },
      attachTo: document.body,
    });
    if (mobile) {
      await wrapper.get('button[aria-label="打开导航"]').trigger("click");
      await settle();
    }
    const surface = new DOMWrapper(mobile ? document.querySelector<HTMLElement>('[role="dialog"]')! : wrapper.get("aside").element);
    const workspace = surface.get('nav[aria-label="工作区会话"]');
    const recent = surface.get('nav[aria-label="最近会话"]');
    expect(workspace.findAll("button").map(b => b.text())).toEqual(["项目 A", "项目 B"]);
    expect(recent.findAll("button").map(b => b.text())).toEqual(["普通 A", "普通 B"]);
    expect(workspace.get('[aria-current="page"]').text()).toBe("项目 A");
    expect(recent.find('[aria-current="page"]').exists()).toBe(false);
    expect(workspace.element.parentElement).toBe(recent.element.parentElement);
    await wrapper.setProps({ selectedConversationId: 3 });
    expect(workspace.find('[aria-current="page"]').exists()).toBe(false);
    expect(recent.get('[aria-current="page"]').text()).toBe("普通 B");
    await wrapper.setProps({ activeView: "memory" });
    expect(recent.find('[aria-current="page"]').exists()).toBe(false);
    await workspace.findAll("button")[1]!.trigger("click");
    await settle();
    expect(wrapper.emitted("select")).toEqual([[4]]);
    if (mobile) expect(document.querySelector('[role="dialog"]')).toBeNull();
    wrapper.unmount();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("opens the mobile navigation sheet and routes its Knowledge action", async () => {
    const wrapper = mount(AppSidebar, {
      props: {
        conversations: [{ id: 1, title: "A", created_at: "x", updated_at: "x", last_message_at: null }],
        selectedConversationId: 1,
        activeView: "chat",
        memoryUpdated: false,
        busy: false,
        loading: false,
      },
      attachTo: document.body,
    });

    expect(wrapper.get('button[aria-label="打开导航"]').exists()).toBe(true);
    await wrapper.get('button[aria-label="打开导航"]').trigger("click");
    await settle();

    const dialog = document.body.querySelector<HTMLElement>('[role="dialog"]');
    expect(dialog?.textContent).toContain("访问聊天、知识库、长期记忆、技能和最近会话。");
    const skillsButton = Array.from(dialog?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent?.includes("技能"));
    expect(skillsButton).toBeDefined();
    skillsButton?.click();
    await settle();

    expect(wrapper.emitted("openSkills")).toHaveLength(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    wrapper.unmount();
  });

  it("shows 运行观测 only when explicitly enabled and preserves normal navigation", async () => {
    const props = {
      conversations: [],
      selectedConversationId: null,
      activeView: "chat" as const,
      memoryUpdated: false,
      busy: false,
      loading: false,
    };
    const production = mount(AppSidebar, { props });
    expect(production.text()).toContain("聊天");
    expect(production.text()).toContain("知识库");
    expect(production.text()).toContain("记忆");
    expect(production.get('nav[aria-label="导航"]').text()).toContain("技能");
    expect(production.text()).not.toContain("开发者");
    expect(production.text()).not.toContain("运行观测");
    production.unmount();

    const development = mount(AppSidebar, {
      props: { ...props, developerEnabled: true },
    });
    const normalNavigation = development.get('nav[aria-label="导航"]');
    const skillsButton = normalNavigation.findAll("button").find((item) => item.text() === "技能")!;
    await skillsButton.trigger("click");
    expect(development.emitted("openSkills")).toHaveLength(1);
    await development.setProps({ activeView: "skills" });
    expect(skillsButton.attributes("aria-current")).toBe("page");
    expect(development.get('nav[aria-label="开发者"]').text()).toContain("运行观测");
    expect(development.get('nav[aria-label="开发者"]').text()).not.toContain("技能");
    expect(development.get('nav[aria-label="开发者"]').text()).not.toContain("LLM 调用");
    await development.get('nav[aria-label="开发者"] button').trigger("click");
    expect(development.emitted("openObservatory")).toHaveLength(1);
    development.unmount();
  });
});
