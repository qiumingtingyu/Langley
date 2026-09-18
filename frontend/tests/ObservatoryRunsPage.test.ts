import { flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

import ObservatoryRunsPage from "../src/ObservatoryRunsPage.vue";

const metrics = {
  percentile_method: "nearest_rank:ceil(p*N),one_based",
  run: {
    indexed_authorized_run_count: 2,
    terminal_observed_run_count: 1,
    business_success_count: 1,
    business_failure_count: 0,
    business_cancelled_count: 1,
    business_terminal_count: 2,
    business_success_rate: 1,
    trace_complete_count: 1,
    trace_complete_rate: 0.5,
    latency: { sample_count: 1, average_ms: 1200, p50_ms: 1200, p95_ms: 1200 },
    provider_tokens: {},
    rounds_per_run_average: 1,
    tools_per_run_average: 0.5,
  },
  llm: {
    event_count: 2,
    latency: { sample_count: 2, average_ms: 500, p50_ms: 400, p95_ms: 600 },
    ttft: { sample_count: 1, average_ms: 84, p50_ms: 84, p95_ms: 84 },
    input_tokens: { observed_total: 242000, observed_round_count: 1, total_round_count: 2, coverage_rate: 0.5 },
    output_tokens: { observed_total: 800, observed_round_count: 2, total_round_count: 2, coverage_rate: 1 },
    total_tokens: { observed_total: 0, observed_round_count: 0, total_round_count: 2, coverage_rate: 0 },
  },
  tool: { scope: "selected_runs", event_count: 1, latency: { sample_count: 1, average_ms: 20, p50_ms: 20, p95_ms: 20 } },
};

const runs = [
  {
    run_id: 42,
    business_status: "SUCCEEDED",
    diagnostics: {
      available: true,
      observed_outcome: "SUCCEEDED",
      trace_complete: false,
      first_observed_timestamp: "2026-09-17T10:00:00Z",
      last_observed_timestamp: "2026-09-17T10:00:01Z",
      provider: "qwen",
      configured_model: "model-a",
      capture_mode: "FULL_CONTENT",
      round_count: 2,
      tool_count: 1,
      provider_input_tokens: null,
      provider_output_tokens: 800,
      provider_total_tokens: null,
      observed_duration_ms: 1200,
    },
  },
  {
    run_id: 41,
    business_status: "CANCELLED",
    diagnostics: {
      available: false,
      observed_outcome: null,
      trace_complete: null,
      first_observed_timestamp: null,
      last_observed_timestamp: null,
      provider: null,
      configured_model: null,
      capture_mode: null,
      round_count: null,
      tool_count: null,
      provider_input_tokens: null,
      provider_output_tokens: null,
      provider_total_tokens: null,
      observed_duration_ms: null,
    },
  },
];

function response(payload: unknown, status = 200): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => payload } as Response;
}

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("ObservatoryRunsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders observation-aware metrics, separate statuses, trends, and the Run inspector seam", async () => {
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/dev/observatory/runs") return Promise.resolve(response({ runs }));
      if (path === "/api/dev/observatory/runs/42") return Promise.resolve(response(runs[0]));
      if (path === "/api/dev/observatory/runs/42/timeline") return Promise.resolve(response({ run_id: 42, events: [] }));
      return Promise.resolve(response({ metrics }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunsPage);
    await settle();

    expect(wrapper.text()).toContain("运行观测台");
    expect(wrapper.text()).toContain("开发者模式 · Run Observatory");
    expect(wrapper.text()).toContain("100.0%");
    expect(wrapper.text()).toContain("1.20 s");
    expect(wrapper.text()).toContain("84 ms");
    expect(wrapper.text()).toContain("242,000 已观测");
    expect(wrapper.text()).toContain("1 / 2 Rounds · 覆盖率 50.0%");
    expect(wrapper.text()).toContain("SUCCEEDED");
    expect(wrapper.text()).toContain("部分");
    expect(wrapper.text()).toContain("CANCELLED");
    expect(wrapper.text()).toContain("未捕获");
    expect(wrapper.text()).toContain("未知");
    expect(wrapper.text()).toContain("观测时长");
    expect(wrapper.text()).toContain("Provider 输入 Token / Run");
    expect(wrapper.text()).toContain("观测时长 / Run");
    const durationTrend = wrapper.get('[data-testid="trend-observed-duration-by-run"]');
    expect(durationTrend.text()).not.toContain("Run latency");
    expect(durationTrend.findAll(".y-axis-tick").length).toBeGreaterThanOrEqual(3);
    expect(durationTrend.findAll(".y-axis-tick").length).toBeLessThanOrEqual(5);
    expect(durationTrend.findAll(".x-tick-label").map(label => label.text())).toEqual(["#42", "#41"]);
    const coverage = wrapper.get('[role="progressbar"][aria-label="LLM 输入 Token 观测覆盖率"]');
    expect(coverage.attributes("aria-valuenow")).toBe("50");
    expect(coverage.get("span").attributes("style")).toContain("width: 50%");
    expect(wrapper.text()).not.toContain("0 ms");
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes("/raw"))).toBe(true);

    await wrapper.get('select[aria-label="业务状态"]').setValue("SUCCEEDED");
    await wrapper.get(".run-link").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("Run 概览");
    expect(wrapper.text()).toContain("#42");
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes("/events/"))).toBe(true);

    await wrapper.get('button[data-slot="button"]').trigger("click");
    expect(wrapper.text()).toContain("运行观测台");
    expect((wrapper.get('select[aria-label="业务状态"]').element as HTMLSelectElement).value).toBe("SUCCEEDED");
  });

  it("renders an empty state without inventing Run data", async () => {
    vi.stubGlobal("fetch", vi.fn((path: string) =>
      Promise.resolve(path === "/api/dev/observatory/runs" ? response({ runs: [] }) : response({ metrics: {
        ...metrics,
        run: { ...metrics.run, indexed_authorized_run_count: 0, business_success_rate: null, trace_complete_rate: null, latency: { sample_count: 0, average_ms: null, p50_ms: null, p95_ms: null } },
        llm: { ...metrics.llm, ttft: { sample_count: 0, average_ms: null, p50_ms: null, p95_ms: null }, input_tokens: { observed_total: 0, observed_round_count: 0, total_round_count: 0, coverage_rate: null } },
      } })),
    ));
    const wrapper = mount(ObservatoryRunsPage);
    await settle();
    expect(wrapper.text()).toContain("暂无已索引 Runs");
    expect(wrapper.text()).toContain("未知");
    expect(wrapper.text()).toContain("0 / 0 Rounds · 覆盖率 未知");
  });

  it.each([
    [503, "开发者观测台暂不可用。"],
    [500, "诊断数据读取失败。"],
  ])("renders the correct failure state for HTTP %s", async (status, message) => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(response({}, status))));
    const wrapper = mount(ObservatoryRunsPage);
    await settle();
    expect(wrapper.text()).toContain(message);
  });

  it("uses only supported model/capture metric filters and filters completeness locally", async () => {
    const fetchMock = vi.fn((path: string) =>
      Promise.resolve(path === "/api/dev/observatory/runs" ? response({ runs }) : response({ metrics })),
    );
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunsPage);
    await settle();

    await wrapper.get('select[aria-label="Model"]').setValue("model-a");
    await settle();
    expect(fetchMock).toHaveBeenCalledWith("/api/dev/observatory/metrics?configured_model=model-a");
    await wrapper.get('input[aria-label="仅完整 Trace"]').setValue(true);
    expect(wrapper.text()).toContain("没有符合当前筛选的 Runs");
    expect(fetchMock.mock.calls.some(([path]) => String(path).includes("trace_complete"))).toBe(false);
  });

  it("orders and bounds trend points while preserving missing values as gaps", async () => {
    const manyRuns = Array.from({ length: 26 }, (_, index) => ({
      ...runs[0],
      run_id: index + 1,
      diagnostics: {
        ...runs[0].diagnostics,
        first_observed_timestamp: index === 4 ? "invalid" : `2026-09-${String(index + 1).padStart(2, "0")}T10:00:00Z`,
        provider_input_tokens: index === 25 ? null : index * 10,
        observed_duration_ms: index === 24 ? null : index * 100,
      },
    }));
    vi.stubGlobal("fetch", vi.fn((path: string) =>
      Promise.resolve(path === "/api/dev/observatory/runs" ? response({ runs: manyRuns }) : response({ metrics })),
    ));
    const wrapper = mount(ObservatoryRunsPage);
    await settle();

    const tokenChart = wrapper.get('[data-testid="trend-provider-input-tokens-by-run"]');
    const accessiblePoints = tokenChart.findAll(".sr-only li");
    expect(accessiblePoints).toHaveLength(24);
    expect(accessiblePoints[0].text()).toContain("Run 3：");
    expect(accessiblePoints.at(-1)?.text()).toContain("Run 5：");
    expect(tokenChart.text()).toContain("Run 26：未捕获");
    expect(tokenChart.find('[data-point-id="26"]').exists()).toBe(false);
    expect(tokenChart.find('[data-point-id="25"]').exists()).toBe(true);
    const yLabels = tokenChart.findAll(".y-tick-label").map(label => label.text());
    expect(yLabels[0]).toBe("0");
    expect(yLabels).toHaveLength(4);
    const xLabels = tokenChart.findAll(".x-tick-label").map(label => label.text());
    expect(xLabels).toEqual(["#3", "#8", "#12", "#16", "#20", "#24", "#5"]);
    expect(xLabels.length).toBeLessThan(accessiblePoints.length);
    await tokenChart.get('[data-point-id="25"]').trigger("mouseenter");
    expect(tokenChart.text()).toContain("Run #25");
    const durationChart = wrapper.get('[data-testid="trend-observed-duration-by-run"]');
    expect(durationChart.findAll(".y-tick-label").map(label => label.text())).toEqual(["0", "1s", "2s", "3s"]);
    expect(durationChart.text()).toContain("Run 25：未捕获");
    expect(durationChart.find('[data-point-id="25"]').exists()).toBe(false);
  });

  it("scopes trends by model and capture mode, but not ledger-only filters", async () => {
    const fetchMock = vi.fn((path: string) =>
      Promise.resolve(path === "/api/dev/observatory/runs" ? response({ runs }) : response({ metrics })),
    );
    vi.stubGlobal("fetch", fetchMock);
    const wrapper = mount(ObservatoryRunsPage);
    await settle();

    const chart = () => wrapper.get('[data-testid="trend-observed-duration-by-run"]');
    expect(wrapper.text()).toContain("模型与捕获模式同时作用于指标、趋势和账本");
    expect(wrapper.text()).toContain("业务状态与 Trace 完整性仅筛选当前账本");
    expect(chart().findAll(".sr-only li")).toHaveLength(2);
    await wrapper.get('select[aria-label="业务状态"]').setValue("CANCELLED");
    expect(chart().findAll(".sr-only li")).toHaveLength(2);
    await wrapper.get('select[aria-label="Model"]').setValue("model-a");
    await settle();
    expect(chart().findAll(".sr-only li")).toHaveLength(1);
    await wrapper.get('select[aria-label="捕获模式"]').setValue("FULL_CONTENT");
    await settle();
    expect(chart().findAll(".sr-only li")).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/dev/observatory/metrics?configured_model=model-a&capture_mode=FULL_CONTENT",
    );
  });
});
