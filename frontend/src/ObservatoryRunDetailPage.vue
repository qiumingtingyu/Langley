<script setup lang="ts">
import { ArrowLeft, ChevronLeft, ChevronRight, Code2, RefreshCw } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";

import ObservatoryContextInspector from "@/components/observatory/ObservatoryContextInspector.vue";
import ObservatoryTrendChart from "@/components/observatory/ObservatoryTrendChart.vue";
import { Button } from "@/components/ui/button";
import {
  aggregateContextByKind,
  diffContextComponents,
  type ObservatoryContextResponse,
  type ObservatoryRawEventResponse,
  type ObservatoryRoundDetail,
  type ObservatoryRun,
  type ObservatoryTimelineEvent,
  type ObservatoryTimelineResponse,
} from "@/observatory";

const props = defineProps<{ runId: number }>();
defineEmits<{ back: [] }>();

const detail = ref<ObservatoryRun | null>(null);
const detailError = ref(false);
const detailLoading = ref(true);
const timeline = ref<ObservatoryTimelineEvent[]>([]);
const timelineError = ref(false);
const timelineLoading = ref(true);
const roundDetails = ref<Record<number, ObservatoryRoundDetail>>({});
const roundFailures = ref(new Set<number>());
const selectedRound = ref<number | null>(null);
const currentContext = ref<ObservatoryContextResponse | null>(null);
const previousContext = ref<ObservatoryContextResponse | null>(null);
const contextLoading = ref(false);
const contextError = ref(false);
const previousContextUnavailable = ref(false);
const rawEvent = ref<ObservatoryRawEventResponse | null>(null);
const rawLoading = ref(false);
const rawError = ref<string | null>(null);
const rawEventId = ref<number | null>(null);
const contextInspectorOpen = ref(false);
const showAllComponents = ref(false);
const showRetainedComponents = ref(false);
let contextRevision = 0;
let rawRevision = 0;

async function readJson<Response>(path: string): Promise<Response> {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: { code?: string } | string } | null;
    const code = typeof payload?.detail === "object" ? payload.detail.code : null;
    throw new Error(code ?? `HTTP_${response.status}`);
  }
  return (await response.json()) as Response;
}

const roundNumbers = computed(() =>
  [...new Set(timeline.value.flatMap(event => (event.round === null ? [] : [event.round])))].sort((a, b) => a - b),
);
const selectedRoundDetail = computed(() =>
  selectedRound.value === null ? null : (roundDetails.value[selectedRound.value] ?? null),
);
const selectedRoundIndex = computed(() =>
  selectedRound.value === null ? -1 : roundNumbers.value.indexOf(selectedRound.value),
);
const previousRoundNumber = computed(() =>
  selectedRoundIndex.value > 0 ? (roundNumbers.value[selectedRoundIndex.value - 1] ?? null) : null,
);
const nextRoundNumber = computed(() =>
  selectedRoundIndex.value >= 0 && selectedRoundIndex.value < roundNumbers.value.length - 1
    ? (roundNumbers.value[selectedRoundIndex.value + 1] ?? null)
    : null,
);
const roundChartPoints = computed(() =>
  roundNumbers.value.map(round => ({ id: round, value: roundDetails.value[round]?.provider_input_tokens ?? null })),
);

function hasProjectedContextFrame(context: ObservatoryContextResponse | null): boolean {
  return context !== null && context.estimate_kind !== null && context.estimated_semantic_context_tokens !== null;
}

const currentContextAvailable = computed(() => hasProjectedContextFrame(currentContext.value));
const previousContextAvailable = computed(() => hasProjectedContextFrame(previousContext.value));
const contextEstimatorsMatch = computed(
  () => currentContextAvailable.value
    && previousContextAvailable.value
    && currentContext.value!.estimate_kind === previousContext.value!.estimate_kind,
);
const contextAggregates = computed(() =>
  aggregateContextByKind(currentContextAvailable.value ? currentContext.value!.components : []),
);
const maximumContextKind = computed(() => Math.max(...contextAggregates.value.map(item => item.estimatedTokens), 1));
const sortedComponents = computed(() =>
  [...(currentContextAvailable.value ? currentContext.value!.components : [])].sort(
    (left, right) => right.estimated_tokens - left.estimated_tokens || left.key.localeCompare(right.key),
  ),
);
const displayedComponents = computed(() =>
  showAllComponents.value ? sortedComponents.value : sortedComponents.value.slice(0, 5),
);
const contextDiff = computed(() =>
  contextEstimatorsMatch.value
    ? diffContextComponents(previousContext.value!.components, currentContext.value!.components).sort((left, right) => {
        const statusOrder = { ADDED: 0, CHANGED: 1, REMOVED: 2, RETAINED: 3 };
        const byStatus = statusOrder[left.status] - statusOrder[right.status];
        const leftMagnitude = Math.max(left.previousTokens ?? 0, left.currentTokens ?? 0);
        const rightMagnitude = Math.max(right.previousTokens ?? 0, right.currentTokens ?? 0);
        return byStatus || rightMagnitude - leftMagnitude || left.key.localeCompare(right.key);
      })
    : [],
);
const changedContextDiff = computed(() => contextDiff.value.filter(item => item.status !== "RETAINED"));
const retainedContextDiff = computed(() => contextDiff.value.filter(item => item.status === "RETAINED"));
const contextDiffCounts = computed(() => ({
  ADDED: contextDiff.value.filter(item => item.status === "ADDED").length,
  CHANGED: contextDiff.value.filter(item => item.status === "CHANGED").length,
  REMOVED: contextDiff.value.filter(item => item.status === "REMOVED").length,
  RETAINED: retainedContextDiff.value.length,
}));
const currentContextTotal = computed(() =>
  currentContextAvailable.value ? currentContext.value!.estimated_semantic_context_tokens : null,
);
const previousContextTotal = computed(() =>
  previousContextAvailable.value ? previousContext.value!.estimated_semantic_context_tokens : null,
);
const contextTotalDelta = computed(() =>
  !contextEstimatorsMatch.value || currentContextTotal.value === null || previousContextTotal.value === null
    ? null
    : currentContextTotal.value - previousContextTotal.value,
);
const usesMeasurementDiff = computed(() => contextDiff.value.some(item => item.comparisonBasis === "measurements"));
const rootEvents = computed(() => timeline.value.filter(event => event.round === null));
const roundEventGroups = computed(() =>
  roundNumbers.value.map(round => ({ round, events: timeline.value.filter(event => event.round === round) })),
);
const selectedRoundLLMEvent = computed(() =>
  selectedRound.value === null
    ? null
    : (timeline.value.find(event => event.kind === "llm" && event.round === selectedRound.value) ?? null),
);

function formatInteger(value: number | null): string {
  return value === null ? "未捕获" : value.toLocaleString("en-US");
}

function formatDuration(value: number | null): string {
  if (value === null) return "未捕获";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toLocaleString("en-US")} ms`;
}

function formatTimestamp(value: string | null): string {
  if (value === null) return "未知";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "未知" : date.toLocaleString("zh-CN", { hour12: false });
}

function diffStatusLabel(status: "ADDED" | "REMOVED" | "CHANGED" | "RETAINED"): string {
  return {
    ADDED: "新增 · ADDED",
    REMOVED: "移除 · REMOVED",
    CHANGED: "变化 · CHANGED",
    RETAINED: "保留 · RETAINED",
  }[status];
}

function comparisonBasisLabel(basis: "content hash" | "measurements"): string {
  return basis === "content hash" ? "内容哈希" : "测量值";
}

function formatMetadata(event: ObservatoryTimelineEvent): string[] {
  return Object.entries(event.metadata).map(([key, value]) => `${key}: ${String(value)}`);
}

async function loadRound(round: number): Promise<void> {
  if (roundDetails.value[round] || roundFailures.value.has(round)) return;
  try {
    const payload = await readJson<ObservatoryRoundDetail>(`/api/dev/observatory/runs/${props.runId}/rounds/${round}`);
    roundDetails.value = { ...roundDetails.value, [round]: payload };
  } catch {
    roundFailures.value = new Set([...roundFailures.value, round]);
  }
}

async function loadContext(round: number): Promise<void> {
  const revision = ++contextRevision;
  contextLoading.value = true;
  contextError.value = false;
  previousContextUnavailable.value = false;
  currentContext.value = null;
  previousContext.value = null;
  const roundIndex = roundNumbers.value.indexOf(round);
  const previousRound = roundIndex > 0 ? (roundNumbers.value[roundIndex - 1] ?? null) : null;
  try {
    const current = await readJson<ObservatoryContextResponse>(`/api/dev/observatory/runs/${props.runId}/rounds/${round}/context`);
    if (revision !== contextRevision) return;
    currentContext.value = current;
  } catch {
    if (revision === contextRevision) contextError.value = true;
  }
  if (previousRound !== null && revision === contextRevision) {
    try {
      const previous = await readJson<ObservatoryContextResponse>(
        `/api/dev/observatory/runs/${props.runId}/rounds/${previousRound}/context`,
      );
      if (revision !== contextRevision) return;
      previousContext.value = previous;
      previousContextUnavailable.value = !hasProjectedContextFrame(previous);
    } catch {
      if (revision === contextRevision) previousContextUnavailable.value = true;
    }
  }
  if (revision === contextRevision) contextLoading.value = false;
}

async function selectRound(round: number): Promise<void> {
  contextInspectorOpen.value = false;
  showAllComponents.value = false;
  showRetainedComponents.value = false;
  selectedRound.value = round;
  rawRevision += 1;
  rawEvent.value = null;
  rawEventId.value = null;
  rawError.value = null;
  rawLoading.value = false;
  await Promise.all([loadRound(round), loadContext(round)]);
}

async function loadRaw(event: ObservatoryTimelineEvent): Promise<void> {
  const revision = ++rawRevision;
  rawEventId.value = event.event_id;
  rawEvent.value = null;
  rawError.value = null;
  rawLoading.value = true;
  try {
    const payload = await readJson<ObservatoryRawEventResponse>(
      `/api/dev/observatory/runs/${props.runId}/events/${event.event_id}/raw`,
    );
    if (revision !== rawRevision) return;
    rawEvent.value = payload;
  } catch (error) {
    if (revision !== rawRevision) return;
    rawError.value = error instanceof Error && error.message === "OBSERVATORY_RAW_EVENT_UNAVAILABLE"
      ? "该 Trace 的原始诊断记录不可用。"
      : "无法加载原始诊断记录。";
  } finally {
    if (revision === rawRevision) rawLoading.value = false;
  }
}

async function loadDetail(): Promise<void> {
  detailLoading.value = true;
  detailError.value = false;
  try {
    detail.value = await readJson<ObservatoryRun>(`/api/dev/observatory/runs/${props.runId}`);
  } catch {
    detailError.value = true;
  } finally {
    detailLoading.value = false;
  }
}

async function loadTimeline(): Promise<void> {
  timelineLoading.value = true;
  timelineError.value = false;
  try {
    const payload = await readJson<ObservatoryTimelineResponse>(`/api/dev/observatory/runs/${props.runId}/timeline`);
    timeline.value = payload.events;
    await Promise.all(roundNumbers.value.map(round => loadRound(round)));
    const initialRound = roundNumbers.value.at(-1);
    if (initialRound !== undefined) await selectRound(initialRound);
  } catch {
    timelineError.value = true;
  } finally {
    timelineLoading.value = false;
  }
}

function reload(): void {
  contextInspectorOpen.value = false;
  void loadDetail();
  void loadTimeline();
}

watch(() => props.runId, reload);
onMounted(reload);
</script>

<template>
  <section
    class="inspector"
    aria-labelledby="inspector-title"
  >
    <header class="inspector-header">
      <div>
        <Button
          variant="ghost"
          size="small"
          @click="$emit('back')"
        >
          <ArrowLeft
            data-icon="inline-start"
            aria-hidden="true"
          /> 返回最近 Runs
        </Button>
        <p class="kicker">
          开发者 / Runs / {{ runId }}
        </p>
        <h1 id="inspector-title">
          Run <span>#{{ runId }}</span>
        </h1>
        <p class="subtitle">
          MySQL 业务权威、Trace 诊断观测与完整性分层呈现
        </p>
      </div>
      <Button
        variant="outline"
        size="small"
        @click="reload"
      >
        <RefreshCw
          data-icon="inline-start"
          aria-hidden="true"
        /> 刷新诊断
      </Button>
    </header>

    <section
      class="panel overview"
      aria-labelledby="overview-title"
    >
      <div class="panel-heading">
        <div>
          <p class="kicker">
            权威边界
          </p><h2 id="overview-title">
            Run 概览
          </h2>
        </div>
      </div>
      <p
        v-if="detailLoading"
        class="section-state"
      >
        正在加载 Run 概览…
      </p>
      <p
        v-else-if="detailError"
        class="section-state warning"
      >
        Run 概览暂不可用，仍可继续检查执行时间线。
      </p>
      <div
        v-else-if="detail"
        class="overview-grid"
      >
        <article data-role="business">
          <span>业务状态 · MySQL 权威</span><strong>{{ detail.business_status }}</strong><small>Run 业务结果的权威事实</small>
        </article>
        <article data-role="diagnostic">
          <span>观测结果 · diagnostics</span><strong>{{ detail.diagnostics.observed_outcome ?? "未知" }}</strong><small>诊断结果可能与业务状态不同</small>
        </article>
        <article data-role="integrity">
          <span>Trace 完整性</span><strong>{{ detail.diagnostics.trace_complete === null ? "未捕获" : detail.diagnostics.trace_complete ? "完整" : "部分" }}</strong><small>仅表示诊断记录质量</small>
        </article>
        <article><span>观测时长</span><strong class="mono">{{ formatDuration(detail.diagnostics.observed_duration_ms) }}</strong><small>已观测 Trace 跨度，可能并非完整 Run 延迟</small></article>
        <article><span>Provider / 配置模型</span><strong class="mono">{{ detail.diagnostics.provider ?? "未知" }}</strong><small>{{ detail.diagnostics.configured_model ?? "模型未知" }}</small></article>
        <article><span>捕获模式</span><strong class="mono">{{ detail.diagnostics.capture_mode ?? "未知" }}</strong><small>本地诊断捕获策略</small></article>
        <article><span>Rounds / Tools</span><strong class="mono">{{ detail.diagnostics.round_count ?? "—" }} / {{ detail.diagnostics.tool_count ?? "—" }}</strong><small>已索引诊断事件</small></article>
        <article><span>Provider 输入</span><strong class="mono">{{ formatInteger(detail.diagnostics.provider_input_tokens) }}</strong><small>严格 Run 聚合</small></article>
        <article><span>Provider 输出</span><strong class="mono">{{ formatInteger(detail.diagnostics.provider_output_tokens) }}</strong><small>严格 Run 聚合</small></article>
        <article><span>Provider 总计</span><strong class="mono">{{ formatInteger(detail.diagnostics.provider_total_tokens) }}</strong><small>任一 Round 缺失即为未知</small></article>
      </div>
    </section>

    <section
      class="panel round-trend-panel"
      aria-labelledby="round-trend-title"
    >
      <div class="panel-heading">
        <div>
          <p class="kicker">
            Provider 观测
          </p><h2 id="round-trend-title">
            Round 输入 Token
          </h2>
        </div>
      </div>
      <p
        v-if="timelineLoading"
        class="section-state"
      >
        正在加载 Round 观测…
      </p>
      <p
        v-else-if="timelineError"
        class="section-state warning"
      >
        执行时间线不可用，无法生成 Round 趋势。
      </p>
      <ObservatoryTrendChart
        v-else
        title="Provider 输入 Token / Round"
        description="各已索引 LLM Round 的 Provider 上报输入用量。"
        entity-label="Round"
        test-id="trend-provider-input-tokens-by-round"
        :points="roundChartPoints"
        value-label="输入 Token"
        :format-value="value => value.toLocaleString('en-US')"
      />
    </section>

    <section
      class="panel timeline-panel"
      aria-labelledby="timeline-title"
    >
      <div class="panel-heading">
        <div>
          <p class="kicker">
            执行记录
          </p><h2 id="timeline-title">
            执行时间线
          </h2>
        </div><span
          v-if="!timelineError"
          class="mono"
        >{{ timeline.length }} 个事件</span>
      </div>
      <p
        v-if="timelineLoading"
        class="section-state"
      >
        正在加载诊断时间线…
      </p>
      <p
        v-else-if="timelineError"
        class="section-state warning"
      >
        执行时间线暂不可用，上方权威概览仍可使用。
      </p>
      <div
        v-else
        class="timeline-groups"
      >
        <article
          v-if="rootEvents.length"
          class="timeline-group timeline-group--root"
        >
          <h3>Run 根事件</h3><ol>
            <li
              v-for="event in rootEvents"
              :key="event.event_id"
              :data-kind="event.kind"
            >
              <div class="event-row">
                <div>
                  <strong class="event-kind">{{ event.kind }}</strong><span class="event-time mono">{{ formatTimestamp(event.timestamp) }}</span><span
                    v-if="event.duration_ms !== null"
                    class="event-duration mono"
                  >{{ formatDuration(event.duration_ms) }}</span><div
                    v-if="formatMetadata(event).length"
                    class="metadata"
                  >
                    <span
                      v-for="item in formatMetadata(event)"
                      :key="item"
                    >{{ item }}</span>
                  </div>
                </div><Button
                  variant="ghost"
                  size="small"
                  :aria-label="`查看原始诊断记录 ${event.event_id}`"
                  @click="loadRaw(event)"
                >
                  <Code2
                    data-icon="inline-start"
                    aria-hidden="true"
                  /> 原始记录
                </Button>
              </div>
            </li>
          </ol>
        </article>
        <article
          v-for="group in roundEventGroups"
          :key="group.round"
          class="timeline-group"
          :data-selected="selectedRound === group.round"
        >
          <button
            class="round-heading"
            type="button"
            @click="selectRound(group.round)"
          >
            <span>Round {{ group.round }}</span><ChevronRight
              :size="15"
              aria-hidden="true"
            />
          </button>
          <ol>
            <li
              v-for="event in group.events"
              :key="event.event_id"
              :data-kind="event.kind"
            >
              <div class="event-row">
                <div>
                  <strong class="event-kind">{{ event.kind }}</strong><span class="event-time mono">{{ formatTimestamp(event.timestamp) }}</span><span
                    v-if="event.duration_ms !== null"
                    class="event-duration mono"
                  >{{ formatDuration(event.duration_ms) }}</span><div
                    v-if="formatMetadata(event).length"
                    class="metadata"
                  >
                    <span
                      v-for="item in formatMetadata(event)"
                      :key="item"
                    >{{ item }}</span>
                  </div>
                </div><Button
                  variant="ghost"
                  size="small"
                  :aria-label="`查看原始诊断记录 ${event.event_id}`"
                  @click="loadRaw(event)"
                >
                  <Code2
                    data-icon="inline-start"
                    aria-hidden="true"
                  /> 原始记录
                </Button>
              </div>
            </li>
          </ol>
        </article>
      </div>
      <aside
        v-if="rawEventId !== null"
        class="raw-viewer"
        aria-live="polite"
      >
        <div><strong>原始诊断记录 · ID #{{ rawEventId }}</strong><small>事件 ID 仅用于定位，不代表时间顺序。</small></div><p class="raw-privacy">
          隐私提示：本地开发 Trace 可能包含会话、Memory、Knowledge、Tool 或 Workspace 内容。
        </p><p v-if="rawLoading">
          正在加载原始诊断记录…
        </p><p
          v-else-if="rawError"
          class="warning"
        >
          {{ rawError }}
        </p><template v-else-if="rawEvent">
          <p
            v-if="rawEvent.capture_mode === 'METADATA_ONLY'"
            class="metadata-note"
          >
            当前为 METADATA_ONLY：正文与内容哈希按设计不会被捕获。
          </p><pre>{{ JSON.stringify(rawEvent.event, null, 2) }}</pre>
        </template>
      </aside>
    </section>

    <section
      class="panel round-panel"
      aria-labelledby="round-title"
    >
      <div class="panel-heading">
        <div>
          <p class="kicker">
            Round 诊断
          </p><h2 id="round-title">
            {{ selectedRound === null ? "尚未选择 Round" : `Round ${selectedRound}` }}
          </h2>
        </div><span
          v-if="selectedRoundDetail"
          class="mono"
        >{{ selectedRoundDetail.provider_model ?? "模型未知" }}</span>
      </div>
      <p
        v-if="timelineError || roundNumbers.length === 0"
        class="section-state"
      >
        暂无可检查的已索引 Round。
      </p>
      <template v-else-if="selectedRound !== null">
        <nav
          class="round-tabs"
          aria-label="可用 Rounds"
        >
          <button
            v-for="round in roundNumbers"
            :key="round"
            type="button"
            :aria-current="selectedRound === round ? 'true' : undefined"
            @click="selectRound(round)"
          >
            R{{ round }}
          </button>
        </nav>
        <div class="context-inspector-action">
          <div class="context-inspector-copy">
            <p class="context-inspector-kicker">
              CONTEXT INSPECTOR
            </p>
            <strong>检查本轮模型上下文</strong>
            <span>查看实际送入模型的 provider-neutral 标准化请求</span>
            <code>
              R{{ selectedRound }} · {{ detail?.diagnostics.capture_mode ?? "捕获模式未知" }} ·
              约 {{ formatInteger(selectedRoundDetail?.estimated_semantic_context_tokens ?? null) }} Token
            </code>
          </div>
          <Button @click="contextInspectorOpen = true">
            查看上下文 →
          </Button>
        </div>
        <p
          v-if="roundFailures.has(selectedRound)"
          class="section-state warning"
        >
          Round 测量暂不可用，时间线事件仍可查看。
        </p>
        <div
          v-else-if="selectedRoundDetail"
          class="measurement-grid"
        >
          <article data-measure="provider">
            <span>Provider 输入</span><strong class="mono">{{ formatInteger(selectedRoundDetail.provider_input_tokens) }}</strong>
          </article>
          <article data-measure="provider">
            <span>Provider 输出</span><strong class="mono">{{ formatInteger(selectedRoundDetail.provider_output_tokens) }}</strong>
          </article>
          <article data-measure="provider">
            <span>Provider 总计</span><strong class="mono">{{ formatInteger(selectedRoundDetail.provider_total_tokens) }}</strong>
          </article>
          <article data-measure="semantic">
            <span>语义上下文估算</span><strong class="mono">{{ formatInteger(selectedRoundDetail.estimated_semantic_context_tokens) }}</strong><small>{{ selectedRoundDetail.estimate_kind ?? "估算器未知" }} · 非 Provider 精确 Token</small>
          </article>
          <article><span>LLM 时长</span><strong class="mono">{{ formatDuration(selectedRoundDetail.duration_ms) }}</strong></article>
          <article><span>TTFT</span><strong class="mono">{{ formatDuration(selectedRoundDetail.ttft_ms) }}</strong></article>
          <article><span>结束原因</span><strong class="mono">{{ selectedRoundDetail.finish_reason ?? "未捕获" }}</strong></article>
        </div>

        <p
          v-if="contextLoading"
          class="section-state"
        >
          正在加载 ContextFrame 投影…
        </p>
        <p
          v-else-if="contextError"
          class="section-state warning"
        >
          本轮 ContextFrame 暂不可用，Provider 与时间线观测仍可使用。
        </p>
        <p
          v-else-if="!currentContextAvailable"
          class="section-state warning"
        >
          本轮未捕获 ContextFrame；Provider 与时间线观测仍可使用。
        </p>
        <template v-else-if="currentContext">
          <div class="context-heading">
            <div>
              <p class="kicker">
                语义上下文
              </p><h3>本轮上下文构成</h3>
              <p class="context-heading-copy">
                各类组件占用的语义 Token 估算
              </p>
            </div><span>{{ currentContext.estimate_kind ?? "估算器未知" }} · 语义估算，并非精确 Token</span>
          </div>
          <div class="context-total">
            <span>本轮语义总估算</span>
            <strong class="mono">约 {{ formatInteger(currentContextTotal) }} Token</strong>
            <small>{{ contextAggregates.length }} 类 · {{ sortedComponents.length }} 个组件</small>
          </div>
          <div class="context-bars">
            <article
              v-for="(item, index) in contextAggregates"
              :key="item.kind"
              :data-largest="index === 0"
            >
              <div>
                <strong><span class="mono context-rank">{{ index + 1 }}</span>{{ item.kind }}</strong>
                <span class="mono">约 {{ formatInteger(item.estimatedTokens) }} Token · {{ item.componentCount }} 个组件 · {{ currentContextTotal ? `${Math.round((item.estimatedTokens / currentContextTotal) * 100)}%` : "占比未知" }}</span>
              </div><i aria-hidden="true"><b :style="{ width: `${(item.estimatedTokens / maximumContextKind) * 100}%` }" /></i>
            </article>
          </div>
          <div class="context-heading context-heading--details">
            <div>
              <p class="kicker">
                Top 5 概览
              </p><h3>组件明细</h3>
            </div><span>按语义 Token 估算从高到低</span>
          </div>
          <div class="table-scroll">
            <table>
              <thead><tr><th>Key</th><th>Kind</th><th>来源</th><th>字符数</th><th>UTF-8 字节</th><th>估算 Token</th><th>内容哈希</th></tr></thead><tbody>
                <tr
                  v-for="component in displayedComponents"
                  :key="component.key"
                >
                  <td class="mono">
                    {{ component.key }}
                  </td><td>{{ component.kind }}</td><td>{{ component.source ?? "—" }}</td><td class="mono">
                    {{ component.chars }}
                  </td><td class="mono">
                    {{ component.utf8_bytes }}
                  </td><td class="mono">
                    {{ component.estimated_tokens }}
                  </td><td class="mono hash">
                    {{ component.content_sha256 ?? "未捕获" }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div
            v-if="sortedComponents.length > 5"
            class="component-detail-toggle"
          >
            <Button
              variant="ghost"
              size="small"
              :aria-expanded="showAllComponents"
              @click="showAllComponents = !showAllComponents"
            >
              {{ showAllComponents ? "收起组件明细" : `查看全部组件 (${sortedComponents.length})` }}
            </Button>
          </div>
        </template>

        <div class="context-heading context-heading--diff">
          <div>
            <p class="kicker">
              Round 对比
            </p><h3>与上一 Round 的变化</h3>
          </div>
          <nav
            class="round-pager"
            aria-label="Context Round 翻页"
          >
            <Button
              variant="ghost"
              size="small"
              aria-label="查看上一 Round"
              :disabled="previousRoundNumber === null"
              @click="previousRoundNumber !== null && selectRound(previousRoundNumber)"
            >
              <ChevronLeft
                :size="15"
                aria-hidden="true"
              />
              上一轮
            </Button>
            <span class="mono">R{{ selectedRound }} · {{ selectedRoundIndex + 1 }}/{{ roundNumbers.length }}</span>
            <Button
              variant="ghost"
              size="small"
              aria-label="查看下一 Round"
              :disabled="nextRoundNumber === null"
              @click="nextRoundNumber !== null && selectRound(nextRoundNumber)"
            >
              下一轮
              <ChevronRight
                :size="15"
                aria-hidden="true"
              />
            </Button>
          </nav>
        </div>
        <p
          v-if="contextLoading"
          class="section-state"
        >
          正在加载本轮对比基线…
        </p>
        <p
          v-else-if="contextError || !currentContextAvailable"
          class="section-state warning"
        >
          当前 Round 的 ContextFrame 不可用，无法计算变化；仍可切换到其他 Round。
        </p>
        <template v-else-if="currentContext">
          <p
            v-if="previousRoundNumber === null"
            class="section-state"
          >
            首个已观测 Round 没有上一轮 ContextFrame 基线。
          </p>
          <p
            v-else-if="previousContextUnavailable || !previousContextAvailable"
            class="section-state warning"
          >
            上一轮 ContextFrame 不可用，无法计算变化。
          </p>
          <p
            v-else-if="!contextEstimatorsMatch"
            class="section-state warning"
          >
            两轮使用的 Context 估算器版本不同，语义 Token 总量与组件变化不可直接比较。
          </p>
          <template v-else-if="previousContext">
            <div
              class="diff-summary"
              aria-label="Context 总量对比"
            >
              <span>上一轮 <strong class="mono">{{ formatInteger(previousContextTotal) }}</strong></span>
              <span>当前轮 <strong class="mono">{{ formatInteger(currentContextTotal) }}</strong></span>
              <span>变化量 <strong class="mono">{{ contextTotalDelta === null ? "未知" : `${contextTotalDelta >= 0 ? "+" : ""}${contextTotalDelta.toLocaleString("en-US")}` }}</strong></span>
            </div>
            <div
              class="diff-counts"
              aria-label="Context 组件变化统计"
            >
              <span data-status="ADDED">新增 <strong class="mono">{{ contextDiffCounts.ADDED }}</strong></span>
              <span data-status="CHANGED">变化 <strong class="mono">{{ contextDiffCounts.CHANGED }}</strong></span>
              <span data-status="REMOVED">移除 <strong class="mono">{{ contextDiffCounts.REMOVED }}</strong></span>
              <span data-status="RETAINED">未变化 <strong class="mono">{{ contextDiffCounts.RETAINED }}</strong></span>
            </div>
            <p
              v-if="usesMeasurementDiff"
              class="comparison-note"
            >
              部分组件没有内容哈希；此时“保留”仅表示 Key 与测量值稳定，不代表已验证正文完全相同。
            </p><div class="diff-grid">
              <article
                v-for="item in changedContextDiff"
                :key="item.key"
                :data-status="item.status"
              >
                <span>{{ diffStatusLabel(item.status) }}</span><strong class="mono">{{ item.key }}</strong><small>{{ item.kind }} · {{ item.previousTokens ?? "—" }} → {{ item.currentTokens ?? "—" }} · {{ comparisonBasisLabel(item.comparisonBasis) }}</small>
              </article>
            </div>
            <div
              v-if="retainedContextDiff.length"
              class="retained-disclosure"
            >
              <Button
                variant="ghost"
                size="small"
                :aria-expanded="showRetainedComponents"
                @click="showRetainedComponents = !showRetainedComponents"
              >
                {{ showRetainedComponents ? "收起未变化组件" : `查看 ${retainedContextDiff.length} 个未变化组件` }}
              </Button>
              <div
                v-if="showRetainedComponents"
                class="diff-grid diff-grid--retained"
              >
                <article
                  v-for="item in retainedContextDiff"
                  :key="item.key"
                  :data-status="item.status"
                >
                  <span>{{ diffStatusLabel(item.status) }}</span><strong class="mono">{{ item.key }}</strong><small>{{ item.kind }} · {{ item.previousTokens ?? "—" }} → {{ item.currentTokens ?? "—" }} · {{ comparisonBasisLabel(item.comparisonBasis) }}</small>
                </article>
              </div>
            </div>
          </template>
        </template>
      </template>
    </section>

    <ObservatoryContextInspector
      v-if="selectedRound !== null"
      v-model:open="contextInspectorOpen"
      :run-id="runId"
      :round="selectedRound"
      :llm-event="selectedRoundLLMEvent"
      :context="currentContextAvailable ? currentContext : null"
      :round-detail="selectedRoundDetail"
      :capture-mode="detail?.diagnostics.capture_mode ?? null"
    />
  </section>
</template>

<style scoped>
.inspector{display:grid;gap:1.25rem}.inspector-header,.panel-heading,.context-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.inspector-header h1,.panel h2,.context-heading h3,p{margin:0}.inspector-header h1{margin-top:.25rem;font-size:clamp(1.7rem,3vw,2.4rem);letter-spacing:-.045em}.inspector-header h1 span{color:var(--primary-deep);font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.kicker{margin-top:.7rem;color:var(--primary-deep);font:600 .68rem/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.12em;text-transform:uppercase}.subtitle{margin-top:.45rem;color:var(--muted-foreground);font-size:.8rem}.panel{overflow:hidden;border:1px solid var(--strong-border);border-radius:var(--radius-lg);background:var(--surface);box-shadow:0 12px 32px rgba(31,45,49,.035)}.panel-heading{align-items:flex-end;padding:1rem 1.1rem;border-bottom:1px solid var(--border)}.panel-heading .kicker,.context-heading .kicker{margin-top:0}.panel h2{margin-top:.2rem;font-size:1.1rem}.panel-heading>span,.context-heading>span{color:var(--muted-foreground);font-size:.68rem}.overview-grid,.measurement-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}.overview-grid article,.measurement-grid article{display:grid;gap:.4rem;padding:1rem 1.1rem;border-right:1px solid var(--border);border-top:1px solid var(--border)}.overview-grid article:nth-child(-n+3){border-top:0}.overview-grid article:nth-child(3n),.measurement-grid article:nth-child(3n){border-right:0}.overview-grid span,.measurement-grid span{color:var(--muted-foreground);font-size:.67rem}.overview-grid strong,.measurement-grid strong{font-size:.95rem}.overview-grid small,.measurement-grid small{color:var(--muted-foreground);font-size:.64rem}.section-state{padding:1.25rem 1.1rem;color:var(--muted-foreground);font-size:.78rem}.warning{color:var(--warning-foreground)}.timeline-groups{padding:.5rem 1.1rem 1.1rem}.timeline-group{border-left:1px solid var(--strong-border);padding-left:1rem}.timeline-group h3,.round-heading{margin:0;padding:.7rem 0;color:var(--foreground);font-size:.75rem}.round-heading{display:flex;width:100%;align-items:center;justify-content:space-between;border:0;background:transparent;font-weight:650}.timeline-group[data-selected=true] .round-heading{color:var(--primary-deep)}.timeline-group ol{margin:0;padding:0;list-style:none}.timeline-group li{position:relative;padding:.55rem 0;border-top:1px solid var(--border)}.timeline-group li:before{position:absolute;left:-1.25rem;top:1rem;width:.46rem;height:.46rem;border:2px solid var(--surface);border-radius:50%;background:var(--primary);content:""}.event-row{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.event-row>div{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem}.event-kind{font:.72rem ui-monospace,SFMono-Regular,Consolas,monospace}.event-time,.event-duration{color:var(--muted-foreground);font-size:.65rem}.metadata{flex-basis:100%;display:flex;flex-wrap:wrap;gap:.3rem}.metadata span{padding:.12rem .3rem;background:var(--subtle);color:var(--muted-foreground);font:.6rem ui-monospace,SFMono-Regular,Consolas,monospace}.raw-inline{border:0;background:transparent;color:var(--muted-foreground);font-size:.68rem}.raw-viewer{margin:0 1.1rem 1.1rem;padding:.8rem;border:1px solid var(--warning-border);background:var(--warning-surface)}.raw-viewer>div{display:flex;justify-content:space-between;gap:1rem}.raw-viewer small,.metadata-note{color:var(--warning-foreground);font-size:.64rem}.raw-viewer pre{max-height:24rem;overflow:auto;margin:.7rem 0 0;padding:.7rem;background:var(--foreground);color:var(--background);font-size:.68rem}.round-tabs{display:flex;gap:.4rem;padding:.8rem 1.1rem;border-bottom:1px solid var(--border)}.round-tabs button{border:1px solid var(--border);border-radius:999px;background:var(--surface);padding:.25rem .55rem;color:var(--muted-foreground);font:.68rem ui-monospace,SFMono-Regular,Consolas,monospace}.round-tabs button[aria-current=true]{border-color:var(--primary);background:var(--accent);color:var(--primary-deep)}.context-heading{align-items:flex-end;padding:1.1rem;border-top:1px solid var(--border)}.context-heading h3{margin-top:.2rem;font-size:.95rem}.context-bars{display:grid;gap:.7rem;padding:0 1.1rem 1.1rem}.context-bars article>div{display:flex;justify-content:space-between;gap:1rem;font-size:.7rem}.context-bars article span{color:var(--muted-foreground)}.context-bars i{display:block;height:3px;margin-top:.3rem;background:var(--border)}.context-bars b{display:block;height:100%;background:var(--primary)}.table-scroll{overflow-x:auto;border-top:1px solid var(--border)}table{width:100%;min-width:58rem;border-collapse:collapse;font-size:.7rem}th,td{padding:.6rem .7rem;text-align:left;border-top:1px solid var(--border)}th{border-top:0;background:var(--subtle);color:var(--muted-foreground);font-size:.6rem;text-transform:uppercase}.hash{max-width:14rem;overflow:hidden;text-overflow:ellipsis}.comparison-note{padding:0 1.1rem .8rem;color:var(--muted-foreground);font-size:.68rem}.diff-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6rem;padding:0 1.1rem 1.1rem}.diff-grid article{display:grid;gap:.18rem;padding:.65rem;border-left:3px solid var(--strong-border);background:var(--subtle)}.diff-grid article[data-status=ADDED]{border-color:var(--success-foreground)}.diff-grid article[data-status=REMOVED]{border-color:var(--danger-foreground)}.diff-grid article[data-status=CHANGED]{border-color:var(--warning-foreground)}.diff-grid span{font-size:.58rem;font-weight:700}.diff-grid small{color:var(--muted-foreground);font-size:.62rem}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;font-variant-numeric:tabular-nums}@media(max-width:760px){.overview-grid,.measurement-grid,.diff-grid{grid-template-columns:1fr}.overview-grid article,.measurement-grid article{border-right:0}.overview-grid article:nth-child(-n+3){border-top:1px solid var(--border)}.overview-grid article:first-child{border-top:0}.inspector-header{display:grid}}
.raw-privacy {
  margin-top: 0.55rem;
  color: var(--warning-foreground);
  font-size: 0.64rem;
}

.diff-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  padding: 0 1.1rem 0.8rem;
  color: var(--muted-foreground);
  font-size: 0.68rem;
}

.diff-summary strong {
  margin-left: 0.3rem;
  color: var(--foreground);
}

.inspector-header {
  position: relative;
  padding: 0.25rem 0 1.15rem 1rem;
  border-bottom: 1px solid var(--strong-border);
}

.inspector-header::before {
  position: absolute;
  top: 0.2rem;
  bottom: 1.15rem;
  left: 0;
  width: 3px;
  border-radius: 999px;
  background: var(--primary);
  content: "";
}

.panel {
  background: color-mix(in srgb, var(--surface) 96%, var(--accent));
  box-shadow:
    inset 0 2px 0 color-mix(in srgb, var(--primary) 34%, transparent),
    0 16px 36px rgba(31, 45, 49, 0.05);
}

.panel-heading {
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--accent) 52%, transparent), transparent 58%),
    var(--surface);
}

.overview {
  box-shadow:
    inset 0 3px 0 color-mix(in srgb, var(--primary) 58%, transparent),
    0 18px 40px rgba(31, 45, 49, 0.055);
}

.overview-grid article[data-role="business"] {
  background: color-mix(in srgb, var(--accent) 52%, var(--surface));
  box-shadow: inset 3px 0 0 var(--success-foreground);
}

.overview-grid article[data-role="diagnostic"] {
  background: color-mix(in srgb, var(--subtle) 70%, var(--surface));
  box-shadow: inset 3px 0 0 var(--primary);
}

.overview-grid article[data-role="integrity"] {
  background: color-mix(in srgb, var(--warning-surface) 48%, var(--surface));
  box-shadow: inset 3px 0 0 var(--warning-border);
}

.timeline-panel {
  background: color-mix(in srgb, var(--surface) 90%, var(--subtle));
}

.timeline-groups {
  display: grid;
  gap: 0.65rem;
}

.timeline-group {
  border-left: 2px solid color-mix(in srgb, var(--primary) 38%, var(--border));
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  background: color-mix(in srgb, var(--surface) 76%, transparent);
  padding: 0 0.75rem 0 1rem;
}

.timeline-group--root {
  border-left-style: dashed;
  background: color-mix(in srgb, var(--subtle) 72%, transparent);
}

.timeline-group[data-selected="true"] {
  border-left-color: var(--primary);
  background: color-mix(in srgb, var(--accent) 64%, var(--surface));
  box-shadow:
    inset 0 0 0 1px color-mix(in srgb, var(--primary) 22%, transparent),
    0 8px 20px rgba(31, 45, 49, 0.045);
}

.timeline-group li::before {
  left: -1.32rem;
  width: 0.56rem;
  height: 0.56rem;
  border-width: 2px;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary) 10%, transparent);
}

.timeline-group li[data-kind="tool"]::before,
.timeline-group li[data-kind="knowledge.search"]::before {
  background: var(--warning-border);
}

.timeline-group li[data-kind="citation.validate"]::before,
.timeline-group li[data-kind="run.success"]::before {
  background: var(--success-foreground);
}

.timeline-group li[data-kind="run.failure"]::before,
.timeline-group li[data-kind="response.rejected"]::before {
  background: var(--danger-foreground);
}

.event-kind {
  color: var(--foreground);
  letter-spacing: 0.015em;
}

.metadata span {
  border: 1px solid color-mix(in srgb, var(--primary) 12%, var(--border));
  border-radius: var(--radius-sm);
}

.raw-viewer {
  border-left: 3px solid var(--warning-border);
  border-radius: var(--radius-md);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--warning-border) 45%, transparent);
}

.round-panel {
  box-shadow:
    inset 0 3px 0 color-mix(in srgb, var(--primary) 64%, transparent),
    0 18px 40px rgba(31, 45, 49, 0.055);
}

.round-tabs {
  background: color-mix(in srgb, var(--subtle) 58%, var(--surface));
}

.round-tabs button[aria-current="true"] {
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--primary) 18%, transparent);
}

.measurement-grid article[data-measure="provider"] {
  background: color-mix(in srgb, var(--subtle) 50%, var(--surface));
}

.measurement-grid article[data-measure="semantic"] {
  background: color-mix(in srgb, var(--accent) 62%, var(--surface));
  box-shadow: inset 3px 0 0 var(--primary);
}

.section-state.warning {
  background: color-mix(in srgb, var(--warning-surface) 68%, var(--surface));
  border-left: 3px solid var(--warning-border);
}

.context-inspector-action {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.75rem;
  padding: 1rem 1.1rem;
  border: 1px solid color-mix(in srgb, var(--primary) 30%, var(--border));
  border-left: 4px solid var(--primary);
  border-radius: var(--radius-lg);
  background: color-mix(in srgb, var(--accent) 58%, var(--surface));
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--primary) 16%, transparent);
}

.context-inspector-copy {
  display: grid;
  gap: 0.22rem;
}

.context-inspector-kicker {
  margin: 0;
  color: var(--primary-deep);
  font-family: var(--langley-font-mono);
  font-size: 0.6875rem;
  font-weight: 650;
  letter-spacing: 0.09em;
}

.context-inspector-action strong {
  font-size: 1rem;
  font-weight: 680;
}

.context-inspector-action span {
  color: var(--muted-foreground);
  font-size: 0.8125rem;
}

.context-inspector-action code {
  color: var(--body);
  font-family: var(--langley-font-mono);
  font-size: 0.75rem;
}

/* Observatory readability floor: prose stays sans; canonical diagnostics stay mono. */
.kicker {
  font-family: var(--langley-font-sans);
  font-size: 0.75rem;
  line-height: 1.35;
  letter-spacing: 0.04em;
  text-transform: none;
}

.subtitle {
  font-size: 0.875rem;
}

.panel h2,
.context-heading h3 {
  font-size: 1.1rem;
  font-weight: 650;
  letter-spacing: -0.015em;
}

.panel-heading > span,
.context-heading > span {
  font-size: 0.8125rem;
}

.overview-grid span,
.measurement-grid span,
.overview-grid small,
.measurement-grid small,
.event-time,
.event-duration,
.raw-inline,
.raw-viewer small,
.metadata-note,
.raw-privacy,
.comparison-note,
.diff-grid span,
.diff-grid small,
.diff-summary {
  font-size: 0.75rem;
}

.section-state,
.timeline-group h3,
.round-heading,
.event-kind,
.context-bars article > div,
table {
  font-size: 0.8125rem;
}

.metadata span,
.round-tabs button,
.raw-viewer pre,
th {
  font-size: 0.75rem;
}

.mono,
.inspector-header h1 span,
.event-kind,
.metadata span,
.round-tabs button,
.raw-viewer pre {
  font-family: var(--langley-font-mono);
}

.context-heading-copy {
  margin-top: 0.3rem;
  color: var(--muted-foreground);
  font-size: 0.8125rem;
}

.context-total {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.2rem 1rem;
  margin: 0 1.1rem 0.9rem;
  padding: 0.85rem 1rem;
  border: 1px solid color-mix(in srgb, var(--primary) 24%, var(--border));
  border-left: 4px solid var(--primary);
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--accent) 48%, var(--surface));
}

.context-total > span,
.context-total > small {
  color: var(--muted-foreground);
  font-size: 0.75rem;
}

.context-total > strong {
  grid-row: span 2;
  align-self: center;
  font-size: 1.05rem;
}

.context-bars article {
  padding: 0.15rem 0;
}

.context-bars article[data-largest="true"] {
  color: var(--primary-deep);
  font-weight: 650;
}

.context-bars article > div strong {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.context-rank {
  display: inline-grid;
  width: 1.35rem;
  height: 1.35rem;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 50%;
  color: var(--muted-foreground);
  font-size: 0.6875rem;
}

.context-bars article[data-largest="true"] .context-rank {
  border-color: color-mix(in srgb, var(--primary) 42%, var(--border));
  background: var(--accent);
  color: var(--primary-deep);
}

.context-bars i {
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--foreground) 9%, transparent);
}

.context-bars b {
  border-radius: inherit;
}

.context-heading--details {
  padding-bottom: 0.85rem;
}

.component-detail-toggle,
.retained-disclosure {
  display: flex;
  justify-content: center;
  padding: 0.7rem 1.1rem;
  border-top: 1px solid var(--border);
  background: color-mix(in srgb, var(--subtle) 48%, var(--surface));
}

.context-heading--diff {
  align-items: center;
}

.round-pager {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.round-pager > span {
  min-width: 5.5rem;
  color: var(--muted-foreground);
  text-align: center;
  font-size: 0.75rem;
}

.diff-counts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  padding: 0 1.1rem 0.8rem;
}

.diff-counts span {
  border: 1px solid var(--border);
  border-left: 3px solid var(--strong-border);
  border-radius: var(--radius-sm);
  padding: 0.3rem 0.5rem;
  color: var(--muted-foreground);
  font-size: 0.75rem;
}

.diff-counts span[data-status="ADDED"] { border-left-color: var(--success-foreground); }
.diff-counts span[data-status="CHANGED"] { border-left-color: var(--warning-foreground); }
.diff-counts span[data-status="REMOVED"] { border-left-color: var(--danger-foreground); }

.diff-counts strong {
  margin-left: 0.25rem;
  color: var(--foreground);
}

.retained-disclosure {
  display: grid;
  justify-items: center;
  padding-top: 0;
  border-top: 0;
  background: transparent;
}

.diff-grid--retained {
  width: 100%;
  padding: 0.7rem 0 0;
}

@media (max-width: 760px) {
  .context-inspector-action {
    align-items: stretch;
    flex-direction: column;
  }

  .context-total {
    grid-template-columns: 1fr;
  }

  .context-total > strong {
    grid-row: auto;
  }

  .context-bars article > div,
  .context-heading--diff,
  .round-pager {
    align-items: flex-start;
  }

  .context-bars article > div,
  .context-heading--diff {
    display: grid;
  }
}
</style>
