import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import MessageContent from "../src/components/MessageContent.vue";

describe("MessageContent", () => {
  it("renders the Markdown baseline and keeps wide content locally scrollable", () => {
    const wrapper = mount(MessageContent, {
      props: {
        content: [
          "# 标题",
          "",
          "**粗体**、*斜体* 与 `inline()`。",
          "",
          "- 无序一",
          "- 无序二",
          "",
          "1. 有序一",
          "2. 有序二",
          "",
          "```ts",
          "const answer = 42;",
          "```",
          "",
          "[外部链接](https://example.com/reference)",
          "",
          "| Source | Status |",
          "| --- | --- |",
          "| Note | Ready |",
        ].join("\n"),
      },
    });

    expect(wrapper.get("h1").text()).toBe("标题");
    expect(wrapper.get("strong").text()).toBe("粗体");
    expect(wrapper.get("em").text()).toBe("斜体");
    expect(wrapper.get("p code").text()).toBe("inline()");
    expect(wrapper.findAll("ul li")).toHaveLength(2);
    expect(wrapper.findAll("ol li")).toHaveLength(2);
    expect(wrapper.get("pre code").text()).toContain("const answer = 42;");
    expect(wrapper.get("pre code").classes()).toContain("language-ts");
    expect(wrapper.get("a").attributes()).toMatchObject({
      href: "https://example.com/reference",
      target: "_blank",
      rel: "noopener noreferrer",
    });
    expect(wrapper.get(".message-table-scroll table").text()).toContain("Ready");
  });

  it("disables raw HTML and unsafe links while preserving citation markers as text", async () => {
    const wrapper = mount(MessageContent, {
      props: {
        content: '<img src="x" onerror="alert(1)">\n\n[危险链接](javascript:alert(1))\n\n证据 [K1]。',
      },
    });

    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.find('a[href^="javascript:"]').exists()).toBe(false);
    expect(wrapper.text()).toContain("<img");
    expect(wrapper.text()).toContain("[K1]");

    await wrapper.setProps({ content: "```js\nconst partial = true;" });
    expect(wrapper.get("pre code").text()).toContain("const partial = true;");
  });

  it("promotes only authoritative citation markers outside code", async () => {
    const citation = {
      evidence_handle: 1,
      document_version_id: 41,
      evidence_text: "authoritative evidence",
      source_display_name: "TCP Notes.md",
      heading_path: ["Transport", "TCP"],
      source_regions: [{ start_byte: 10, end_byte: 32 }],
    };
    const wrapper = mount(MessageContent, {
      props: {
        content: [
          "真实证据 [K1]，缺失证据 [K9]，inline code `[K1]`。",
          "",
          "[K1](https://example.com/citation-label)",
          "",
          "```text",
          "fenced [K1]",
          "```",
        ].join("\n"),
        citations: [citation],
      },
    });

    const citationTriggers = wrapper.findAll('button[data-citation-handle="1"]');
    expect(citationTriggers).toHaveLength(1);
    expect(citationTriggers[0]?.text()).toBe("[1]");
    expect(wrapper.text()).toContain("[K9]");
    expect(wrapper.get("p code").text()).toBe("[K1]");
    expect(wrapper.get("pre code").text()).toContain("fenced [K1]");
    expect(wrapper.get('a[href="https://example.com/citation-label"]').text()).toBe("K1");

    await citationTriggers[0]?.trigger("click");
    expect(wrapper.emitted("selectCitation")?.[0]?.[0]).toEqual(citation);
  });

  it("does not render Markdown images as externally loadable elements", () => {
    const wrapper = mount(MessageContent, {
      props: {
        content: [
          "![tracking](https://example.com/pixel.png)",
          "",
          "**普通 Markdown 保持可用**",
          "",
          "[普通链接](https://example.com/reference)",
        ].join("\n"),
      },
    });

    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.find("[src], [srcset]").exists()).toBe(false);
    expect(wrapper.find("picture, source, image, input[type='image']").exists()).toBe(false);
    expect(wrapper.text()).toContain("tracking");
    expect(wrapper.get("strong").text()).toBe("普通 Markdown 保持可用");
    expect(wrapper.findAll("a").some((link) => link.text() === "普通链接")).toBe(true);
  });
});
