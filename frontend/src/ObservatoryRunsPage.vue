<script setup lang="ts">
import { RefreshCw } from "lucide-vue-next";
import { computed, onMounted, ref, watch } from "vue";

import ObservatoryRunDetailPage from "@/ObservatoryRunDetailPage.vue";
import ObservatoryTrendChart from "@/components/observatory/ObservatoryTrendChart.vue";
import type {
  ObservatoryMetrics,
  ObservatoryMetricsResponse,
  ObservatoryRun,
  ObservatoryRunListResponse,
} from "@/observatory";

type LoadFailure = "unavailable" | "failed";

class ObservatoryRequestError extends Error {
  constructor(readonly status: number) {
    super(`Observatory request failed with status ${status}`);
  }
}

const runs = ref<ObservatoryRun[]>([]);
const metrics = ref<ObservatoryMetrics | null>(null);
const loading = ref(true);
const metricsLoading = ref(false);
const failure = ref<LoadFailure | null>(null);
const selectedRunId = ref<number | null>(null);
const businessStatus = ref("");
const model = ref("");
const captureMode = ref("");
const completeOnly = ref(false);
let metricsRevision = 0;

const modelOptions = computed(() =>
  [...new Set(runs.value.map(run => run.diagnostics.configured_model).filter((value): value is string => value !== null))].sort(),
);
const captureModeOptions = computed(() =>
  [...new Set(runs.value.map(run => run.diagnostics.capture_mode).filter((value): value is string => value !== null))].sort(),
);
const filteredRuns = computed(() =>
  [...runs.value]
    .filter(run => businessStatus.value === "" || run.business_status === businessStatus.value)
    .filter(run => model.value === "" || run.diagnostics.configured_model === model.value)
    .filter(run => captureMode.value === "" || run.diagnostics.capture_mode === captureMode.value)
    .filter(run => !completeOnly.value || run.diagnostics.trace_complete === true)
    .sort((left, right) => {
      const byTime = (right.diagnostics.first_observed_timestamp ?? "").localeCompare(
        left.diagnostics.first_observed_timestamp ?? "",
      );
      return byTime || right.run_id - left.run_id;
    }),
);
const chartRuns = computed(() =>
  [...runs.value]
    .filter(run => model.value === "" || run.diagnostics.configured_model === model.value)
    .filter(run => captureMode.value === "" || run.diagnostics.capture_mode === captureMode.value)
    .sort((left, right) => {
      const parsedLeft = Date.parse(left.diagnostics.first_observed_timestamp ?? "");
      const parsedRight = Date.parse(right.diagnostics.first_observed_timestamp ?? "");
      const leftTime = Number.isNaN(parsedLeft) ? null : parsedLeft;
      const rightTime = Number.isNaN(parsedRight) ? null : parsedRight;
      if (leftTime !== null && rightTime === null) return -1;
      if (leftTime === null && rightTime !== null) return 1;
      if (leftTime !== null && rightTime !== null && leftTime !== rightTime) return leftTime - rightTime;
      return left.run_id - right.run_id;
    })
    .slice(-24),
);
const providerInputPoints = computed(() =>
  chartRuns.value.map(run => ({ id: run.run_id, value: run.diagnostics.provider_input_tokens })),
);
const observedDurationPoints = computed(() =>
  chartRuns.value.map(run => ({ id: run.run_id, value: run.diagnostics.observed_duration_ms })),
);

async function readJson<Response>(path: string): Promise<Response> {
  const response = await fetch(path);
  if (!response.ok) throw new ObservatoryRequestError(response.status);
  return (await response.json()) as Response;
}

function classifyFailure(error: unknown): LoadFailure {
  return error instanceof ObservatoryRequestError && (error.status === 404 || error.status === 503)
    ? "unavailable"
    : "failed";
}

function metricsPath(): string {
  const query = new URLSearchParams();
  if (model.value !== "") query.set("configured_model", model.value);
  if (captureMode.value !== "") query.set("capture_mode", captureMode.value);
  const encoded = query.toString();
  return `/api/dev/observatory/metrics${encoded === "" ? "" : `?${encoded}`}`;
}

async function loadDashboard(): Promise<void> {
  loading.value = true;
  failure.value = null;
  try {
    const [runPayload, metricsPayload] = await Promise.all([
      readJson<ObservatoryRunListResponse>("/api/dev/observatory/runs"),
      readJson<ObservatoryMetricsResponse>(metricsPath()),
    ]);
    runs.value = runPayload.runs;
    metrics.value = metricsPayload.metrics;
  } catch (error) {
    runs.value = [];
    metrics.value = null;
    failure.value = classifyFailure(error);
  } finally {
    loading.value = false;
  }
}

async function loadMetrics(): Promise<void> {
  if (failure.value !== null || loading.value) return;
  const revision = ++metricsRevision;
  metricsLoading.value = true;
  try {
    const payload = await readJson<ObservatoryMetricsResponse>(metricsPath());
    if (revision === metricsRevision) metrics.value = payload.metrics;
  } catch (error) {
    if (revision === metricsRevision) {
      metrics.value = null;
      failure.value = classifyFailure(error);
    }
  } finally {
    if (revision === metricsRevision) metricsLoading.value = false;
  }
}

function formatInteger(value: number | null): string {
  return value === null ? "未知" : value.toLocaleString("en-US");
}

function formatRate(value: number | null): string {
  return value === null ? "未知" : `${(value * 100).toFixed(1)}%`;
}

function formatDuration(value: number | null): string {
  if (value === null) return "未知";
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)} s`;
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 1 })} ms`;
}

function formatTimestamp(value: string | null): string {
  if (value === null) return "未知";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "未知"
    : parsed.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
}

function traceLabel(run: ObservatoryRun): string {
  if (!run.diagnostics.available || run.diagnostics.trace_complete === null) return "未捕获";
  return run.diagnostics.trace_complete ? "完整" : "部分";
}

function resetFilters(): void {
  businessStatus.value = "";
  model.value = "";
  captureMode.value = "";
  completeOnly.value = false;
}

watch([model, captureMode], () => void loadMetrics());
onMounted(() => void loadDashboard());
</script>

<template>
  <section
    class="observatory-shell"
    aria-labelledby="observatory-title"
  >
    <ObservatoryRunDetailPage
      v-if="selectedRunId !== null"
      :run-id="selectedRunId"
      @back="selectedRunId = null"
    />

    <template v-else>
      <header class="observatory-header">
        <div>
          <p class="section-kicker">
            开发者模式 · Run Observatory
          </p>
          <h1 id="observatory-title">
            运行观测台
          </h1>
          <p class="subtitle">
            本地开发诊断 · Trace 数据为尽力观测，不改变业务事实
          </p>
        </div>
        <button
          class="refresh-button"
          type="button"
          :disabled="loading"
          @click="loadDashboard"
        >
          <RefreshCw
            :size="15"
            aria-hidden="true"
          />
          刷新观测
        </button>
      </header>

      <div
        v-if="loading"
        class="state-panel"
        role="status"
      >
        <span class="pulse-dot" />
        正在读取本地诊断…
      </div>

      <div
        v-else-if="failure === 'unavailable'"
        class="state-panel state-panel--warning"
        role="status"
      >
        <strong>开发者观测台暂不可用。</strong>
        <span>后端开发路由未启用，Langley 的正常功能不受影响。</span>
      </div>

      <div
        v-else-if="failure === 'failed'"
        class="state-panel state-panel--warning"
        role="alert"
      >
        <strong>诊断数据读取失败。</strong>
        <span>请检查本地后端后重试；本次操作没有修改任何 Run 数据。</span>
        <button
          type="button"
          @click="loadDashboard"
        >
          重试
        </button>
      </div>

      <template v-else>
        <section
          class="metrics-strip"
          aria-label="运行观测指标"
          :aria-busy="metricsLoading"
        >
          <article>
            <span>已索引 Runs</span>
            <strong class="mono">{{ metrics?.run.indexed_authorized_run_count ?? "未知" }}</strong>
            <small>已通过业务归属校验</small>
          </article>
          <article>
            <span>业务成功率</span>
            <strong>{{ formatRate(metrics?.run.business_success_rate ?? null) }}</strong>
            <small>统计不含 CANCELLED</small>
          </article>
          <article>
            <span>Run 延迟 P95</span>
            <strong class="mono">{{ formatDuration(metrics?.run.latency.p95_ms ?? null) }}</strong>
            <small>{{ metrics?.run.latency.sample_count ?? 0 }} 个完整起止样本</small>
          </article>
          <article>
            <span>LLM 首 Token 时间 P50</span>
            <strong class="mono">{{ formatDuration(metrics?.llm.ttft.p50_ms ?? null) }}</strong>
            <small>{{ metrics?.llm.ttft.sample_count ?? 0 }} 个已观测 Rounds</small>
          </article>
          <article>
            <span>Trace 完整率</span>
            <strong>{{ formatRate(metrics?.run.trace_complete_rate ?? null) }}</strong>
            <small>诊断记录完整性</small>
          </article>
          <article class="token-observation">
            <span>LLM 输入 Token</span>
            <strong class="mono">{{ formatInteger(metrics?.llm.input_tokens.observed_total ?? null) }} 已观测</strong>
            <small v-if="metrics">
              {{ metrics.llm.input_tokens.observed_round_count }} / {{ metrics.llm.input_tokens.total_round_count }} Rounds
              · 覆盖率 {{ formatRate(metrics.llm.input_tokens.coverage_rate) }}
            </small>
            <small v-else>覆盖率未知</small>
            <div
              v-if="metrics"
              class="coverage-progress"
              role="progressbar"
              aria-label="LLM 输入 Token 观测覆盖率"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="Math.round((metrics.llm.input_tokens.coverage_rate ?? 0) * 100)"
            >
              <span :style="{ width: `${(metrics.llm.input_tokens.coverage_rate ?? 0) * 100}%` }" />
            </div>
          </article>
        </section>

        <section
          class="trend-grid"
          aria-label="Run 诊断趋势"
        >
          <ObservatoryTrendChart
            title="Provider 输入 Token / Run"
            description="Provider 上报输入用量的严格 Run 聚合。"
            entity-label="Run"
            test-id="trend-provider-input-tokens-by-run"
            :points="providerInputPoints"
            value-label="输入 Token"
            :format-value="value => value.toLocaleString('en-US')"
          />
          <ObservatoryTrendChart
            title="观测时长 / Run"
            description="首个到末个已观测 Trace 事件的诊断跨度，可能只覆盖部分执行。"
            entity-label="Run"
            test-id="trend-observed-duration-by-run"
            :points="observedDurationPoints"
            value-label="观测时长"
            :format-value="formatDuration"
          />
        </section>

        <section
          class="runs-ledger"
          aria-labelledby="recent-runs-title"
        >
          <div class="ledger-heading">
            <div>
              <p class="section-kicker">
                Trace 账本
              </p>
              <h2 id="recent-runs-title">
                最近 Runs
              </h2>
            </div>
            <span class="mono">{{ filteredRuns.length }} / {{ runs.length }}</span>
          </div>

          <div
            class="filters"
            aria-label="Run 筛选器"
          >
            <label>
              <span>业务状态</span>
              <select
                v-model="businessStatus"
                aria-label="业务状态"
              >
                <option value="">全部状态</option>
                <option
                  v-for="status in ['PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED']"
                  :key="status"
                  :value="status"
                >
                  {{ status }}
                </option>
              </select>
            </label>
            <label>
              <span>Model</span>
              <select
                v-model="model"
                aria-label="Model"
              >
                <option value="">全部模型</option>
                <option
                  v-for="item in modelOptions"
                  :key="item"
                  :value="item"
                >{{ item }}</option>
              </select>
            </label>
            <label>
              <span>捕获模式</span>
              <select
                v-model="captureMode"
                aria-label="捕获模式"
              >
                <option value="">全部模式</option>
                <option
                  v-for="item in captureModeOptions"
                  :key="item"
                  :value="item"
                >{{ item }}</option>
              </select>
            </label>
            <label class="checkbox-filter">
              <input
                v-model="completeOnly"
                type="checkbox"
                aria-label="仅完整 Trace"
              >
              <span>仅完整 Trace</span>
            </label>
            <button
              v-if="businessStatus || model || captureMode || completeOnly"
              type="button"
              class="clear-filter"
              @click="resetFilters"
            >
              清除筛选
            </button>
          </div>
          <p class="filter-note">
            模型与捕获模式同时作用于指标、趋势和账本；业务状态与 Trace 完整性仅筛选当前账本。
          </p>

          <div
            v-if="runs.length === 0"
            class="empty-ledger"
          >
            <strong>暂无已索引 Runs</strong>
            <span>观测到回答模型调用后，本地 Trace 会显示在这里。</span>
          </div>
          <div
            v-else-if="filteredRuns.length === 0"
            class="empty-ledger"
          >
            <strong>没有符合当前筛选的 Runs</strong>
            <button
              type="button"
              @click="resetFilters"
            >
              清除筛选
            </button>
          </div>
          <div
            v-else
            class="table-scroll"
          >
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>业务状态</th>
                  <th>Trace</th>
                  <th>Model</th>
                  <th>Rounds</th>
                  <th>Tools</th>
                  <th>输入 Token</th>
                  <th>观测时长</th>
                  <th>首次观测</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="run in filteredRuns"
                  :key="run.run_id"
                >
                  <td>
                    <button
                      class="run-link mono"
                      type="button"
                      @click="selectedRunId = run.run_id"
                    >
                      #{{ run.run_id }}
                    </button>
                  </td>
                  <td>
                    <span
                      class="status-badge"
                      :data-status="run.business_status"
                    >{{ run.business_status }}</span>
                  </td>
                  <td>
                    <span
                      class="trace-state"
                      :data-state="traceLabel(run)"
                    >{{ traceLabel(run) }}</span>
                  </td>
                  <td class="mono">
                    {{ run.diagnostics.configured_model ?? "未知" }}
                  </td>
                  <td class="mono">
                    {{ run.diagnostics.round_count ?? "—" }}
                  </td>
                  <td class="mono">
                    {{ run.diagnostics.tool_count ?? "—" }}
                  </td>
                  <td class="mono">
                    {{ run.diagnostics.provider_input_tokens === null ? "未捕获" : formatInteger(run.diagnostics.provider_input_tokens) }}
                  </td>
                  <td class="mono">
                    {{ run.diagnostics.observed_duration_ms === null ? "—" : formatDuration(run.diagnostics.observed_duration_ms) }}
                  </td>
                  <td class="mono timestamp">
                    {{ formatTimestamp(run.diagnostics.first_observed_timestamp) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </template>
    </template>
  </section>
</template>

<style scoped>
.observatory-shell {
  min-height: 100%;
  padding: clamp(1.25rem, 3vw, 2.75rem);
  color: var(--foreground);
  background:
    radial-gradient(circle at 1px 1px, color-mix(in srgb, var(--primary) 14%, transparent) 1px, transparent 1.2px),
    linear-gradient(color-mix(in srgb, var(--foreground) 2.5%, transparent) 1px, transparent 1px),
    var(--workspace);
  background-size: 1.5rem 1.5rem, 100% 6rem, auto;
}

.observatory-header,
.ledger-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
}

.observatory-header {
  position: relative;
  padding: 0.25rem 0 1.15rem 1rem;
  border-bottom: 1px solid var(--strong-border);
}

.observatory-header::before {
  position: absolute;
  top: 0.2rem;
  bottom: 1.15rem;
  left: 0;
  width: 3px;
  border-radius: 999px;
  background: var(--primary);
  content: "";
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  margin-top: 0.25rem;
  font-size: clamp(1.7rem, 3vw, 2.4rem);
  letter-spacing: -0.045em;
}

h2 {
  margin-top: 0.2rem;
  font-size: 1.125rem;
  font-weight: 650;
  letter-spacing: -0.02em;
}

.section-kicker {
  color: var(--primary-deep);
  font-family: var(--langley-font-sans);
  font-size: 0.75rem;
  font-weight: 650;
  line-height: 1.35;
  letter-spacing: 0.04em;
}

.subtitle,
.filter-note {
  margin-top: 0.45rem;
  color: var(--muted-foreground);
  font-size: 0.875rem;
}

.mono {
  font-family: var(--langley-font-mono);
  font-variant-numeric: tabular-nums;
}

.refresh-button,
.back-button,
.clear-filter,
.empty-ledger button,
.state-panel button {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-md);
  background: var(--surface);
  color: var(--body);
  padding: 0.52rem 0.72rem;
  font-size: 0.8125rem;
}

.refresh-button:hover,
.back-button:hover,
.run-link:hover {
  color: var(--primary-deep);
  border-color: var(--primary);
}

.refresh-button:focus-visible,
.back-button:focus-visible,
.run-link:focus-visible,
select:focus-visible,
input:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--ring) 55%, transparent);
  outline-offset: 2px;
}

.metrics-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(8.5rem, 1fr));
  margin-top: 2rem;
  overflow-x: auto;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-lg);
  background: color-mix(in srgb, var(--surface) 90%, var(--accent));
  box-shadow:
    inset 0 2px 0 color-mix(in srgb, var(--primary) 52%, transparent),
    0 12px 30px rgba(31, 45, 49, 0.045);
}

.metrics-strip article {
  min-width: 8.5rem;
  padding: 1rem;
  border-right: 1px solid var(--border);
}

.metrics-strip article:last-child {
  border-right: 0;
}

.metrics-strip span,
.metrics-strip small {
  display: block;
  color: var(--muted-foreground);
  font-size: 0.75rem;
}

.metrics-strip strong {
  display: block;
  margin: 0.55rem 0 0.35rem;
  font-size: 1.08rem;
  font-weight: 620;
}

.coverage-progress {
  display: block;
  height: 4px;
  margin-top: 0.45rem;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--primary) 16%, var(--border));
}

.coverage-progress span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--primary);
}

.runs-ledger {
  margin-top: 2.25rem;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow:
    inset 0 2px 0 color-mix(in srgb, var(--primary) 42%, transparent),
    0 18px 42px rgba(31, 45, 49, 0.06);
}

.trend-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.25rem;
}

.ledger-heading {
  padding: 1rem 1.1rem;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--subtle) 58%, var(--surface));
}

.filters {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 0.75rem;
  padding: 0.9rem 1.1rem 0;
}

.filters label:not(.checkbox-filter) {
  display: grid;
  gap: 0.3rem;
  min-width: 9.5rem;
}

.filters label > span {
  color: var(--muted-foreground);
  font-size: 0.75rem;
  letter-spacing: 0.01em;
}

select {
  height: 2rem;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-sm);
  background: var(--workspace);
  color: var(--body);
  padding: 0 1.8rem 0 0.55rem;
  font-size: 0.8125rem;
}

.checkbox-filter {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  height: 2rem;
  color: var(--body);
  font-size: 0.8125rem;
}

.filter-note {
  padding: 0 1.1rem 0.9rem;
  font-size: 0.75rem;
}

.table-scroll {
  overflow-x: auto;
  border-top: 1px solid var(--border);
}

table {
  width: 100%;
  min-width: 70rem;
  border-collapse: collapse;
  font-size: 0.8125rem;
}

th {
  padding: 0.65rem 0.75rem;
  color: var(--muted-foreground);
  background: var(--subtle);
  font-size: 0.75rem;
  font-weight: 550;
  letter-spacing: 0.05em;
  text-align: left;
  text-transform: uppercase;
}

td {
  padding: 0.72rem 0.75rem;
  border-top: 1px solid var(--border);
  color: var(--body);
  white-space: nowrap;
}

tbody tr:hover {
  background: color-mix(in srgb, var(--accent) 58%, transparent);
  box-shadow: inset 3px 0 0 color-mix(in srgb, var(--primary) 68%, transparent);
}

.run-link {
  border: 0;
  border-bottom: 1px solid transparent;
  background: transparent;
  color: var(--foreground);
  padding: 0;
  font-weight: 650;
  text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--primary) 35%, transparent);
  text-underline-offset: 0.22rem;
}

.status-badge,
.trace-state {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.18rem 0.42rem;
  font: 600 0.75rem/1.2 var(--langley-font-mono);
}

.status-badge[data-status="SUCCEEDED"],
.trace-state[data-state="完整"] {
  color: var(--success-foreground);
  background: color-mix(in srgb, var(--accent) 58%, transparent);
}

.status-badge[data-status="FAILED"] {
  color: var(--danger-foreground);
  background: var(--danger-surface);
}

.status-badge[data-status="CANCELLED"],
.trace-state[data-state="部分"] {
  color: var(--warning-foreground);
  background: var(--warning-surface);
}

.trace-state[data-state="未捕获"] {
  color: var(--muted-foreground);
  border-style: dashed;
}

.timestamp {
  color: var(--muted-foreground);
}

.empty-ledger,
.state-panel,
.detail-placeholder {
  display: grid;
  justify-items: start;
  gap: 0.45rem;
  padding: 2rem 1.1rem;
  color: var(--muted-foreground);
  font-size: 0.875rem;
}

.state-panel,
.detail-placeholder {
  margin-top: 2rem;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-lg);
  background: var(--surface);
}

.state-panel--warning {
  border-color: var(--warning-border);
  background: var(--warning-surface);
  color: var(--warning-foreground);
}

.pulse-dot {
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 999px;
  background: var(--primary);
}

.detail-placeholder h1 span {
  color: var(--primary-deep);
}

@media (max-width: 900px) {
  .observatory-header {
    align-items: flex-start;
  }

  .metrics-strip {
    grid-template-columns: repeat(6, minmax(10rem, 1fr));
  }

  .trend-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .observatory-shell {
    padding: 1rem;
  }

  .observatory-header {
    display: grid;
  }

  .refresh-button {
    justify-self: start;
  }
}
</style>
