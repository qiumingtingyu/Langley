import { flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it } from "vitest";

import AppSidebar from "../src/components/AppSidebar.vue";

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("AppSidebar", () => {
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
    expect(dialog?.textContent).toContain("访问聊天、知识库、长期记忆和最近会话。");
    const knowledgeButton = Array.from(dialog?.querySelectorAll<HTMLButtonElement>("button") ?? []).find((item) => item.textContent?.includes("知识库"));
    expect(knowledgeButton).toBeDefined();
    knowledgeButton?.click();
    await settle();

    expect(wrapper.emitted("openKnowledge")).toHaveLength(1);
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    wrapper.unmount();
  });
});
