import type { ObservatoryContextComponent, ObservatoryRawEventResponse } from "@/observatory";

export type ContextBodyStatus = "available" | "metadata_only" | "request_unavailable" | "unsupported";

export interface RenderedContextBody {
  status: ContextBodyStatus;
  value: unknown;
  text: string | null;
  structured: boolean;
  reason: string | null;
}

const KIND_LABELS: Record<string, string> = {
  system: "系统提示",
  personal_context: "个人上下文",
  conversation_compact: "会话摘要",
  "skill.active": "当前 Skill",
  "skill.catalog": "Skill 目录",
  "skill.resources": "Skill 资源",
  "tools.schema": "Tool Schema",
  "transcript.user": "用户消息",
  "transcript.assistant": "助手消息",
  "transcript.tool_call": "Tool 调用",
  "transcript.tool_result": "Tool 结果",
  evidence_context: "证据上下文",
  "unclassified.request_field": "未分类请求字段",
};

/**
 * This adapter intentionally follows the frozen ContextFrame v1 key contract.
 * Unknown or future keys stay visible but are never guessed from display text or measurements.
 */
export function renderContextBody(
  component: ObservatoryContextComponent,
  raw: ObservatoryRawEventResponse | null,
): RenderedContextBody {
  if (raw?.capture_mode === "METADATA_ONLY") return unavailable("metadata_only", "正文未捕获 · METADATA_ONLY");
  const request = raw === null ? null : asRecord(raw.event.request);
  if (request === null) return unavailable("request_unavailable", "本地捕获的标准化请求中没有可用正文");

  const mapped = mapV1Key(component.key, request);
  if (!mapped.supported) return unavailable("unsupported", "正文映射暂未支持");
  if (mapped.value === undefined || mapped.value === null) {
    return unavailable("request_unavailable", "标准化请求中未找到对应正文");
  }
  const structured = typeof mapped.value !== "string";
  return {
    status: "available",
    value: mapped.value,
    text: structured ? JSON.stringify(mapped.value, null, 2) : String(mapped.value),
    structured,
    reason: null,
  };
}

export function contextKindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? "未知上下文组件";
}

export function contextProvenance(kind: string): string {
  if (kind === "system") return "system";
  if (kind === "personal_context") return "personal";
  if (kind === "conversation_compact") return "compact";
  if (kind.startsWith("skill.")) return "skill";
  if (kind === "tools.schema") return "tools";
  if (kind === "transcript.user") return "user";
  if (kind === "transcript.assistant") return "assistant";
  if (kind === "transcript.tool_call") return "tool-call";
  if (kind === "transcript.tool_result") return "tool-result";
  if (kind === "evidence_context") return "evidence";
  if (kind === "unclassified.request_field") return "unclassified";
  return "unknown";
}

function unavailable(status: Exclude<ContextBodyStatus, "available">, reason: string): RenderedContextBody {
  return { status, value: null, text: null, structured: false, reason };
}

function mapV1Key(key: string, request: Record<string, unknown>): { supported: boolean; value?: unknown } {
  if (key === "system") return supported(request.system_input);
  if (key === "conversation_compact") return supported(request.conversation_compact_context);
  if (key === "skill.active") return supported(request.active_skill);
  if (key === "evidence_context") return supported(request.evidence_context);

  let match = /^personal_context\.(\d+)$/.exec(key);
  if (match) return supported(at(request.personal_context, Number(match[1])));
  match = /^skill\.catalog\.(\d+)$/.exec(key);
  if (match) return supported(at(request.available_skills, Number(match[1])));
  match = /^skill\.resources\.(\d+)$/.exec(key);
  if (match) return supported(at(request.active_skill_resources, Number(match[1])));
  match = /^tools\.schema\.(\d+)$/.exec(key);
  if (match) return supported(at(request.allowed_tools, Number(match[1])));
  match = /^transcript\.(\d+)\.(user|assistant)$/.exec(key);
  if (match) return supported(asRecord(at(request.transcript, Number(match[1])))?.content);
  match = /^transcript\.(\d+)\.tool_call\.(\d+)$/.exec(key);
  if (match) {
    const transcriptItem = asRecord(at(request.transcript, Number(match[1])));
    return supported(at(transcriptItem?.tool_calls, Number(match[2])));
  }
  match = /^transcript\.(\d+)\.tool_result$/.exec(key);
  if (match) return supported(at(request.transcript, Number(match[1])));
  return { supported: false };
}

function supported(value: unknown): { supported: true; value: unknown } {
  return { supported: true, value };
}

function at(value: unknown, index: number): unknown {
  return Array.isArray(value) ? value[index] : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}
