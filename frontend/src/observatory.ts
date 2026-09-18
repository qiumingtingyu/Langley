export interface ObservatoryDiagnostics {
  available: boolean;
  observed_outcome: "SUCCEEDED" | "FAILED" | null;
  trace_complete: boolean | null;
  first_observed_timestamp: string | null;
  last_observed_timestamp: string | null;
  provider: string | null;
  configured_model: string | null;
  capture_mode: string | null;
  round_count: number | null;
  tool_count: number | null;
  provider_input_tokens: number | null;
  provider_output_tokens: number | null;
  provider_total_tokens: number | null;
  observed_duration_ms: number | null;
}

export interface ObservatoryRun {
  run_id: number;
  business_status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";
  diagnostics: ObservatoryDiagnostics;
}

export interface DistributionMetrics {
  sample_count: number;
  average_ms: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface LLMTokenObservationMetrics {
  observed_total: number;
  observed_round_count: number;
  total_round_count: number;
  coverage_rate: number | null;
}

export interface ObservatoryMetrics {
  run: {
    indexed_authorized_run_count: number;
    business_success_rate: number | null;
    trace_complete_rate: number | null;
    latency: DistributionMetrics;
  };
  llm: {
    event_count: number;
    ttft: DistributionMetrics;
    input_tokens: LLMTokenObservationMetrics;
    output_tokens: LLMTokenObservationMetrics;
    total_tokens: LLMTokenObservationMetrics;
  };
}

export interface ObservatoryRunListResponse {
  runs: ObservatoryRun[];
}

export interface ObservatoryMetricsResponse {
  metrics: ObservatoryMetrics;
}

export type ObservatoryEventMetadata = Record<string, unknown>;

export interface ObservatoryTimelineEvent {
  event_id: number;
  schema_version: number | null;
  observed_seq: number | null;
  kind: string;
  round: number | null;
  timestamp: string | null;
  duration_ms: number | null;
  tool_call_id: string | null;
  parent_tool_call_id: string | null;
  tool_ordinal: number | null;
  metadata: ObservatoryEventMetadata;
}

export interface ObservatoryTimelineResponse {
  run_id: number;
  events: ObservatoryTimelineEvent[];
}

export interface ObservatoryRoundDetail {
  run_id: number;
  round: number;
  provider_input_tokens: number | null;
  provider_output_tokens: number | null;
  provider_total_tokens: number | null;
  duration_ms: number | null;
  ttft_ms: number | null;
  finish_reason: string | null;
  provider_model: string | null;
  estimate_kind: string | null;
  estimated_semantic_context_tokens: number | null;
}

export interface ObservatoryContextComponent {
  key: string;
  kind: string;
  source: string | null;
  chars: number;
  utf8_bytes: number;
  estimated_tokens: number;
  content_sha256: string | null;
}

export interface ObservatoryContextResponse {
  run_id: number;
  round: number;
  estimate_kind: string | null;
  estimated_semantic_context_tokens: number | null;
  components: ObservatoryContextComponent[];
}

export interface ObservatoryRawEventResponse {
  run_id: number;
  event_id: number;
  capture_mode: string | null;
  event: Record<string, unknown>;
}

export interface ContextKindAggregate {
  kind: string;
  estimatedTokens: number;
  componentCount: number;
}

export type ContextDiffStatus = "ADDED" | "REMOVED" | "CHANGED" | "RETAINED";

export interface ContextComponentDiff {
  key: string;
  kind: string;
  status: ContextDiffStatus;
  previousTokens: number | null;
  currentTokens: number | null;
  comparisonBasis: "content hash" | "measurements";
}

export function aggregateContextByKind(components: ObservatoryContextComponent[]): ContextKindAggregate[] {
  const totals = new Map<string, ContextKindAggregate>();
  for (const component of components) {
    const current = totals.get(component.kind) ?? {
      kind: component.kind,
      estimatedTokens: 0,
      componentCount: 0,
    };
    current.estimatedTokens += component.estimated_tokens;
    current.componentCount += 1;
    totals.set(component.kind, current);
  }
  return [...totals.values()].sort(
    (left, right) => right.estimatedTokens - left.estimatedTokens || left.kind.localeCompare(right.kind),
  );
}

function measurementSignature(component: ObservatoryContextComponent): string {
  return [component.kind, component.source, component.chars, component.utf8_bytes, component.estimated_tokens].join("|");
}

export function diffContextComponents(
  previous: ObservatoryContextComponent[],
  current: ObservatoryContextComponent[],
): ContextComponentDiff[] {
  const previousByKey = new Map(previous.map(component => [component.key, component]));
  const currentByKey = new Map(current.map(component => [component.key, component]));
  const keys = [...new Set([...previousByKey.keys(), ...currentByKey.keys()])].sort();

  return keys.map(key => {
    const before = previousByKey.get(key);
    const after = currentByKey.get(key);
    if (!before) {
      return {
        key,
        kind: after!.kind,
        status: "ADDED",
        previousTokens: null,
        currentTokens: after!.estimated_tokens,
        comparisonBasis: after!.content_sha256 ? "content hash" : "measurements",
      };
    }
    if (!after) {
      return {
        key,
        kind: before.kind,
        status: "REMOVED",
        previousTokens: before.estimated_tokens,
        currentTokens: null,
        comparisonBasis: before.content_sha256 ? "content hash" : "measurements",
      };
    }

    const hashesAvailable = before.content_sha256 !== null && after.content_sha256 !== null;
    const unchanged = hashesAvailable
      ? before.content_sha256 === after.content_sha256
      : measurementSignature(before) === measurementSignature(after);
    return {
      key,
      kind: after.kind,
      status: unchanged ? "RETAINED" : "CHANGED",
      previousTokens: before.estimated_tokens,
      currentTokens: after.estimated_tokens,
      comparisonBasis: hashesAvailable ? "content hash" : "measurements",
    };
  });
}
