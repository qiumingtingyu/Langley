<script setup lang="ts">
import { ChevronDown } from "lucide-vue-next";
import {
  CollapsibleContent,
  CollapsibleRoot,
  CollapsibleTrigger,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from "reka-ui";
import { computed, ref, watch } from "vue";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type {
  ObservatoryContextComponent,
  ObservatoryContextResponse,
  ObservatoryRawEventResponse,
  ObservatoryRoundDetail,
  ObservatoryTimelineEvent,
} from "@/observatory";
import { contextKindLabel, contextProvenance, renderContextBody } from "@/observatoryContext";

const props = defineProps<{
  open: boolean;
  runId: number;
  round: number | null;
  llmEvent: ObservatoryTimelineEvent | null;
  context: ObservatoryContextResponse | null;
  roundDetail: ObservatoryRoundDetail | null;
  captureMode: string | null;
}>();

const emit = defineEmits<{ "update:open": [value: boolean] }>();
const raw = ref<ObservatoryRawEventResponse | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);
const expandedComponentKeys = ref(new Set<string>());
let requestRevision = 0;

const components = computed(() => props.context?.components ?? []);
const effectiveRaw = computed<ObservatoryRawEventResponse | null>(() => {
  if (raw.value !== null) return raw.value;
  if (props.captureMode === "METADATA_ONLY") {
    return { run_id: props.runId, event_id: props.llmEvent?.event_id ?? -1, capture_mode: "METADATA_ONLY", event: {} };
  }
  return null;
});

watch(
  () => [props.open, props.runId, props.round, props.llmEvent?.event_id] as const,
  ([open]) => {
    const revision = ++requestRevision;
    expandedComponentKeys.value = new Set();
    raw.value = null;
    error.value = null;
    loading.value = false;
    if (!open) return;
    if (props.llmEvent === null) {
      error.value = "当前 Round 没有匹配的 LLM 事件，无法定位正文。";
      return;
    }
    loading.value = true;
    void fetch(`/api/dev/observatory/runs/${props.runId}/events/${props.llmEvent.event_id}/raw`)
      .then(async response => {
        if (!response.ok) throw new Error("RAW_UNAVAILABLE");
        return await response.json() as ObservatoryRawEventResponse;
      })
      .then(payload => {
        if (revision === requestRevision) raw.value = payload;
      })
      .catch(() => {
        if (revision === requestRevision) error.value = "本轮原始诊断记录不可用，仍可查看 ContextFrame 测量。";
      })
      .finally(() => {
        if (revision === requestRevision) loading.value = false;
      });
  },
);

function onOpenChange(value: boolean): void {
  if (!value) requestRevision += 1;
  emit("update:open", value);
}

function setComponentExpanded(key: string, open: boolean): void {
  const next = new Set(expandedComponentKeys.value);
  if (open) next.add(key);
  else next.delete(key);
  expandedComponentKeys.value = next;
}

function bodyFor(component: ObservatoryContextComponent) {
  return renderContextBody(component, effectiveRaw.value);
}

function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined ? "未知" : value.toLocaleString("en-US");
}
</script>

<template>
  <Sheet
    :open="open"
    @update:open="onOpenChange"
  >
    <SheetContent
      side="right"
      class="context-sheet-content w-[96vw] min-w-0 gap-0 border-strong-border bg-surface p-0 text-foreground shadow-lg sm:max-w-[44rem] lg:max-w-[48rem]"
    >
      <SheetHeader class="context-sheet-header">
        <p class="context-sheet-kicker">
          模型请求上下文（provider-neutral）
        </p>
        <SheetTitle class="context-sheet-title">
          Round {{ round ?? "—" }} 上下文
        </SheetTitle>
        <SheetDescription class="context-sheet-description">
          {{ roundDetail?.provider_model ?? "模型未知" }} · {{ captureMode ?? "捕获模式未知" }} ·
          约 {{ formatInteger(context?.estimated_semantic_context_tokens) }} 语义 Token 估算
        </SheetDescription>
      </SheetHeader>

      <p class="context-privacy">
        本地 FULL_CONTENT Trace 可能包含会话、Memory、Knowledge、Tool、Skill 与 Workspace 内容。
      </p>
      <p
        v-if="llmEvent"
        class="context-locator"
      >
        原始诊断记录 ID #{{ llmEvent.event_id }} 仅用于定位，不代表执行时间顺序。
      </p>

      <TabsRoot
        default-value="rendered"
        class="context-tabs"
      >
        <TabsList
          class="context-tab-list"
          aria-label="上下文视图"
        >
          <TabsTrigger
            value="rendered"
            class="context-tab-trigger"
          >
            渲染视图
          </TabsTrigger>
          <TabsTrigger
            value="raw"
            class="context-tab-trigger"
          >
            Raw JSON
          </TabsTrigger>
        </TabsList>

        <TabsContent
          value="rendered"
          class="context-sheet-tab-content context-sheet-tab-content--rendered"
        >
          <div class="context-reader-summary">
            <span>{{ components.length }} 个组件</span>
            <span>{{ context?.estimate_kind ?? "估算器未知" }}</span>
            <span>语义估算，不是 Provider 精确归因</span>
          </div>
          <div class="context-reader-body">
            <p
              v-if="loading"
              class="context-reader-state"
            >
              正在加载本地捕获的标准化请求…
            </p>
            <p
              v-if="error"
              class="context-reader-state context-reader-state--warning"
            >
              {{ error }}
            </p>
            <p
              v-if="components.length === 0"
              class="context-reader-state"
            >
              本轮没有可投影的 ContextFrame 组件。
            </p>

            <div class="context-component-list">
              <CollapsibleRoot
                v-for="component in components"
                :key="component.key"
                v-slot="{ open: componentOpen }"
                class="context-component"
                :data-provenance="contextProvenance(component.kind)"
                :open="expandedComponentKeys.has(component.key)"
                @update:open="setComponentExpanded(component.key, $event)"
              >
                <div class="context-component-header">
                  <div>
                    <div class="context-component-title-row">
                      <strong>{{ contextKindLabel(component.kind) }}</strong>
                      <code>{{ component.kind }}</code>
                    </div>
                    <p>
                      <span class="mono">{{ component.key }}</span>
                      <span v-if="component.source"> · {{ component.source }}</span>
                      · 约 {{ formatInteger(component.estimated_tokens) }} tokens
                      · {{ formatInteger(component.chars) }} chars
                      · {{ formatInteger(component.utf8_bytes) }} UTF-8 bytes
                    </p>
                  </div>
                  <CollapsibleTrigger class="context-collapse-trigger">
                    {{ componentOpen ? "收起" : "展开" }}
                    <ChevronDown
                      :size="15"
                      aria-hidden="true"
                    />
                  </CollapsibleTrigger>
                </div>
                <CollapsibleContent class="context-component-content">
                  <p
                    v-if="loading"
                    class="context-body-unavailable"
                  >
                    正在加载正文…
                  </p>
                  <p
                    v-else-if="bodyFor(component).status !== 'available'"
                    class="context-body-unavailable"
                  >
                    {{ bodyFor(component).reason }}
                  </p>
                  <pre
                    v-else-if="bodyFor(component).structured"
                    class="context-structured-body"
                  >{{ bodyFor(component).text }}</pre>
                  <div
                    v-else
                    class="context-text-body"
                  >
                    {{ bodyFor(component).text }}
                  </div>
                </CollapsibleContent>
              </CollapsibleRoot>
            </div>
          </div>
        </TabsContent>

        <TabsContent
          value="raw"
          class="context-sheet-tab-content context-sheet-tab-content--raw"
        >
          <p
            v-if="loading"
            class="context-reader-state"
          >
            正在加载 Raw JSON…
          </p>
          <p
            v-else-if="error"
            class="context-reader-state context-reader-state--warning"
          >
            {{ error }}
          </p>
          <p
            v-else-if="raw?.capture_mode === 'METADATA_ONLY'"
            class="context-reader-state context-reader-state--warning"
          >
            正文未捕获 · METADATA_ONLY。以下仅为可用的原始诊断元数据。
          </p>
          <pre
            v-if="raw"
            class="context-raw-json"
          >{{ JSON.stringify(raw.event, null, 2) }}</pre>
        </TabsContent>
      </TabsRoot>
    </SheetContent>
  </Sheet>
</template>

<style scoped>
.context-sheet-content{height:100%;overflow:hidden;background:color-mix(in srgb,var(--surface) 97%,var(--accent));box-shadow:-24px 0 72px rgba(15,24,28,.2)}
.context-sheet-header{flex-shrink:0;gap:0;padding:1.25rem 3.5rem .9rem 1.35rem;border-bottom:1px solid var(--border);box-shadow:inset 0 3px 0 color-mix(in srgb,var(--primary) 62%,transparent);text-align:left}
.context-sheet-kicker,.context-sheet-title,.context-sheet-description{margin:0}.context-sheet-kicker{color:var(--primary-deep);font-size:.75rem;font-weight:650;letter-spacing:.035em}.context-sheet-title{margin-top:.22rem;font-size:1.35rem;font-weight:680;letter-spacing:-.025em}.context-sheet-description{margin-top:.35rem;color:var(--muted-foreground);font-size:.8125rem}.context-tab-trigger:focus-visible,.context-collapse-trigger:focus-visible{outline:2px solid color-mix(in srgb,var(--ring) 60%,transparent);outline-offset:2px}
.context-privacy,.context-locator{margin:0;padding:.55rem 1.35rem;color:var(--warning-foreground);background:color-mix(in srgb,var(--warning-surface) 72%,transparent);font-size:.75rem}.context-locator{padding-top:0;background:color-mix(in srgb,var(--warning-surface) 72%,transparent);font-family:var(--langley-font-mono);color:var(--muted-foreground)}
.context-tabs{display:flex;min-height:0;flex:1;flex-direction:column;overflow:hidden}.context-tab-list{display:flex;flex-shrink:0;gap:.2rem;padding:.65rem 1.35rem 0;border-bottom:1px solid var(--border);background:color-mix(in srgb,var(--subtle) 45%,transparent)}.context-tab-trigger{border:0;border-bottom:2px solid transparent;background:transparent;padding:.55rem .75rem;color:var(--muted-foreground);font-size:.8125rem;font-weight:600}.context-tab-trigger[data-state=active]{border-bottom-color:var(--primary);color:var(--foreground)}.context-sheet-tab-content{min-height:0;flex:1;overflow-x:hidden;overflow-y:auto;scrollbar-width:thin;scrollbar-color:color-mix(in srgb,var(--primary) 58%,var(--strong-border)) transparent}.context-sheet-tab-content[data-state=inactive]{display:none}.context-sheet-tab-content--raw{padding:1rem 1.35rem 1.4rem}.context-sheet-tab-content::-webkit-scrollbar{width:.65rem}.context-sheet-tab-content::-webkit-scrollbar-track{background:transparent}.context-sheet-tab-content::-webkit-scrollbar-thumb{border:2px solid transparent;border-radius:999px;background:color-mix(in srgb,var(--primary) 58%,var(--strong-border));background-clip:padding-box}.context-reader-summary{display:flex;flex-wrap:wrap;gap:.45rem;padding:.65rem 1.35rem;border-bottom:1px solid var(--border)}.context-reader-summary span{border:1px solid var(--border);border-radius:999px;padding:.18rem .45rem;color:var(--muted-foreground);font-size:.75rem}.context-reader-body{padding:1rem 1.35rem 1.4rem}
.context-reader-state{margin:0 0 .8rem;padding:.75rem;border-left:3px solid var(--strong-border);background:var(--subtle);color:var(--muted-foreground);font-size:.8125rem}.context-reader-state--warning{border-color:var(--warning-border);background:var(--warning-surface);color:var(--warning-foreground)}.context-component-list{display:grid;gap:.8rem}.context-component{--context-rail:var(--primary);overflow:hidden;border:1px solid color-mix(in srgb,var(--context-rail) 34%,var(--border));border-left:5px solid var(--context-rail);border-radius:var(--radius-lg);background:color-mix(in srgb,var(--context-rail) 5%,var(--surface))}.context-component[data-provenance=system]{--context-rail:#3f8291}.context-component[data-provenance=personal]{--context-rail:#b28232}.context-component[data-provenance=compact]{--context-rail:#78858a}.context-component[data-provenance=skill]{--context-rail:#7c6aa6}.context-component[data-provenance=tools]{--context-rail:#596f9d}.context-component[data-provenance=user]{--context-rail:#4f78a8}.context-component[data-provenance=assistant]{--context-rail:#667174}.context-component[data-provenance=tool-call]{--context-rail:#85609a}.context-component[data-provenance=tool-result]{--context-rail:#3e837e}.context-component[data-provenance=evidence]{--context-rail:#4e8467}.context-component[data-provenance=unclassified]{--context-rail:var(--warning-foreground);border-style:dashed}.context-component[data-provenance=unknown]{--context-rail:var(--muted-light);border-color:var(--border);background:color-mix(in srgb,var(--subtle) 45%,var(--surface))}
.context-component-header{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;padding:.8rem .9rem;background:color-mix(in srgb,var(--context-rail) 7%,var(--surface))}.context-component-title-row{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}.context-component-title-row strong{font-size:.9375rem}.context-component-title-row code{border:1px solid color-mix(in srgb,var(--context-rail) 42%,var(--border));border-radius:999px;padding:.12rem .4rem;background:color-mix(in srgb,var(--context-rail) 10%,transparent);color:var(--body);font-size:.7rem}.context-component-header p{margin:.3rem 0 0;color:var(--muted-foreground);font-size:.75rem}.context-collapse-trigger{display:inline-flex;flex-shrink:0;align-items:center;gap:.25rem;border:1px solid var(--border);border-radius:var(--radius-sm);background:transparent;padding:.3rem .45rem;color:var(--muted-foreground);font-size:.75rem}.context-collapse-trigger[data-state=open] svg{transform:rotate(180deg)}.context-component-content{border-top:1px solid color-mix(in srgb,var(--context-rail) 22%,var(--border))}.context-text-body{padding:1rem;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--body);font-size:.9rem;line-height:1.7}.context-structured-body,.context-raw-json{margin:0;padding:1rem;white-space:pre-wrap;overflow-wrap:anywhere;background:color-mix(in srgb,var(--foreground) 96%,#000);color:var(--background);font:400 .8rem/1.65 var(--langley-font-mono)}.context-body-unavailable{margin:0;padding:1rem;color:var(--muted-foreground);font-size:.8125rem}
@media(max-width:760px){.context-sheet-header{padding:1rem 3.5rem .8rem 1rem}.context-privacy,.context-locator,.context-tab-list,.context-reader-summary{padding-left:1rem;padding-right:1rem}.context-reader-body,.context-sheet-tab-content--raw{padding-left:1rem;padding-right:1rem}.context-component-header{display:grid}}
</style>
