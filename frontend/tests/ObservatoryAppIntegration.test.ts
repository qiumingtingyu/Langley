import { flushPromises, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../src/App.vue";

class FakeEventSource {
  onerror: ((event: Event) => void) | null = null;
  addEventListener(): void {}
  close(): void {}
}

function response(payload: unknown): Response {
  return { ok: true, status: 200, json: async () => payload } as Response;
}

async function settle(): Promise<void> {
  await flushPromises();
  await nextTick();
  await flushPromises();
}

describe("Developer Observatory app integration", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("keeps normal navigation and opens 运行观测 from the development-only section", async () => {
    const fetchMock = vi.fn((path: string) => {
      if (path === "/api/conversations" || path === "/api/knowledge-bases") return Promise.resolve(response([]));
      if (path === "/api/dev/observatory/runs") return Promise.resolve(response({ runs: [] }));
      if (path === "/api/dev/observatory/metrics") {
        return Promise.resolve(response({
          metrics: {
            run: {
              indexed_authorized_run_count: 0,
              business_success_rate: null,
              trace_complete_rate: null,
              latency: { sample_count: 0, average_ms: null, p50_ms: null, p95_ms: null },
            },
            llm: {
              event_count: 0,
              ttft: { sample_count: 0, average_ms: null, p50_ms: null, p95_ms: null },
              input_tokens: { observed_total: 0, observed_round_count: 0, total_round_count: 0, coverage_rate: null },
              output_tokens: { observed_total: 0, observed_round_count: 0, total_round_count: 0, coverage_rate: null },
              total_tokens: { observed_total: 0, observed_round_count: 0, total_round_count: 0, coverage_rate: null },
            },
          },
        }));
      }
      throw new Error(`unexpected fetch: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("EventSource", FakeEventSource);
    const wrapper = mount(App);
    await settle();

    expect(wrapper.get('nav[aria-label="导航"]').text()).toContain("聊天");
    expect(wrapper.get('nav[aria-label="导航"]').text()).toContain("知识库");
    expect(wrapper.get('nav[aria-label="导航"]').text()).toContain("记忆");
    const developerNavigation = wrapper.get('nav[aria-label="开发者"]');
    expect(developerNavigation.text()).toContain("运行观测");
    expect(developerNavigation.text()).not.toContain("LLM 调用");

    await developerNavigation.get("button").trigger("click");
    await settle();
    expect(wrapper.text()).toContain("运行观测台");
    expect(wrapper.text()).toContain("暂无已索引 Runs");
    expect(fetchMock.mock.calls.every(([path]) => !String(path).includes("/raw"))).toBe(true);
  });
});
