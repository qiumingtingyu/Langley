import { describe, expect, it } from "vitest";

import type { ObservatoryContextComponent, ObservatoryRawEventResponse } from "../src/observatory";
import { contextKindLabel, contextProvenance, renderContextBody } from "../src/observatoryContext";
import inspectorSource from "../src/components/observatory/ObservatoryContextInspector.vue?raw";

const raw: ObservatoryRawEventResponse = {
  run_id: 42,
  event_id: 5,
  capture_mode: "FULL_CONTENT",
  event: {
    kind: "llm",
    request: {
      system_input: "system body",
      personal_context: ["memory body"],
      conversation_compact_context: "compact body",
      active_skill: { name: "active", instructions: "skill body" },
      available_skills: [{ name: "catalog", description: "catalog body" }],
      active_skill_resources: [{ path: "reference.md", byte_size: 17 }],
      allowed_tools: [{ name: "search", arguments_schema: { type: "object" } }],
      transcript: [
        { role: "user", content: "user body" },
        { role: "assistant", content: "assistant body", tool_calls: [{ name: "search", raw_arguments: "{}" }] },
        { name: "search", kind: "SUCCESS", content: "tool result body" },
      ],
      evidence_context: "evidence body",
    },
  },
};

function component(key: string, kind: string): ObservatoryContextComponent {
  return { key, kind, source: null, chars: 1, utf8_bytes: 1, estimated_tokens: 1, content_sha256: null };
}

describe("ContextFrame v1 body adapter", () => {
  it("protects the Sheet and native-scroll CSS contract without claiming browser scroll behavior", () => {
    expect(inspectorSource).toContain("<Sheet");
    expect(inspectorSource).toContain("<SheetContent");
    expect(inspectorSource).toContain('side="right"');
    expect(inspectorSource).toContain("sm:max-w-[44rem] lg:max-w-[48rem]");
    expect(inspectorSource).not.toContain("DialogRoot");
    expect(inspectorSource).not.toContain("context-dialog-content");
    expect(inspectorSource).not.toContain("ScrollAreaRoot");
    expect(inspectorSource).not.toContain("ScrollAreaViewport");
    expect(inspectorSource).not.toContain("ScrollAreaScrollbar");
    expect(inspectorSource).not.toContain("ScrollAreaThumb");
    expect(inspectorSource).toContain('class="context-sheet-tab-content context-sheet-tab-content--rendered"');
    expect(inspectorSource).toContain('class="context-sheet-tab-content context-sheet-tab-content--raw"');
    expect(inspectorSource).toContain(".context-tabs{display:flex;min-height:0;flex:1;flex-direction:column;");
    expect(inspectorSource).toContain(".context-tab-list{display:flex;flex-shrink:0;");
    expect(inspectorSource).toContain(".context-sheet-tab-content{min-height:0;flex:1;overflow-x:hidden;overflow-y:auto;");
    expect(inspectorSource).not.toContain("height:calc(");
  });

  it("uses explicit controlled multi-component expansion without size heuristics", () => {
    expect(inspectorSource).toContain("const expandedComponentKeys = ref(new Set<string>())");
    expect(inspectorSource).toContain(":open=\"expandedComponentKeys.has(component.key)\"");
    expect(inspectorSource).toContain("@update:open=\"setComponentExpanded(component.key, $event)\"");
    expect(inspectorSource).not.toContain("shouldStartCollapsed");
    expect(inspectorSource).not.toContain(":default-open=");
  });

  it.each([
    ["system", "system", "system body"],
    ["personal_context.0", "personal_context", "memory body"],
    ["conversation_compact", "conversation_compact", "compact body"],
    ["transcript.0.user", "transcript.user", "user body"],
    ["transcript.1.assistant", "transcript.assistant", "assistant body"],
    ["evidence_context", "evidence_context", "evidence body"],
  ])("maps %s without fuzzy matching", (key, kind, expected) => {
    expect(renderContextBody(component(key, kind), raw)).toMatchObject({ status: "available", text: expected, structured: false });
  });

  it.each([
    ["skill.active", "skill.active", "skill body"],
    ["skill.catalog.0", "skill.catalog", "catalog body"],
    ["skill.resources.0", "skill.resources", "reference.md"],
    ["tools.schema.0", "tools.schema", "arguments_schema"],
    ["transcript.1.tool_call.0", "transcript.tool_call", "raw_arguments"],
    ["transcript.2.tool_result", "transcript.tool_result", "tool result body"],
  ])("pretty-renders structured %s", (key, kind, expected) => {
    const result = renderContextBody(component(key, kind), raw);
    expect(result.status).toBe("available");
    expect(result.structured).toBe(true);
    expect(result.text).toContain(expected);
    expect(result.text).toContain("\n");
  });

  it("keeps future components visible without guessing a body", () => {
    const future = component("future.component.0", "future.context");
    expect(contextKindLabel(future.kind)).toBe("未知上下文组件");
    expect(contextProvenance(future.kind)).toBe("unknown");
    expect(renderContextBody(future, raw)).toMatchObject({ status: "unsupported", reason: "正文映射暂未支持" });
  });

  it("reports metadata-only and unexpectedly absent request bodies without fabricating content", () => {
    expect(renderContextBody(component("system", "system"), { ...raw, capture_mode: "METADATA_ONLY", event: {} }))
      .toMatchObject({ status: "metadata_only", reason: "正文未捕获 · METADATA_ONLY" });
    expect(renderContextBody(component("system", "system"), { ...raw, event: { kind: "llm" } }))
      .toMatchObject({ status: "request_unavailable" });
  });
});
