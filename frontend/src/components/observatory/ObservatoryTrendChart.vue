<script setup lang="ts">
import { computed, ref } from "vue";

export interface ObservatoryTrendPoint {
  id: number;
  value: number | null;
}

const props = defineProps<{
  title: string;
  description: string;
  entityLabel: "Run" | "Round";
  testId?: string;
  points: ObservatoryTrendPoint[];
  valueLabel: string;
  formatValue: (value: number) => string;
}>();

const activeId = ref<number | null>(null);
const width = 640;
const height = 210;
const plot = { left: 52, right: 18, top: 16, bottom: 36 };
const observedValues = computed(() => props.points.flatMap(point => (point.value === null ? [] : [point.value])));

function niceStep(value: number): number {
  const roughStep = Math.max(value, 1) / 3;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

const yStep = computed(() => niceStep(Math.max(...observedValues.value, 1)));
const yMaximum = computed(() => yStep.value * Math.ceil(Math.max(...observedValues.value, 1) / yStep.value));
const yTicks = computed(() =>
  Array.from({ length: Math.round(yMaximum.value / yStep.value) + 1 }, (_, index) => index * yStep.value),
);
const xTicks = computed(() => {
  const lastIndex = props.points.length - 1;
  if (lastIndex < 0) return [];
  const stride = props.points.length <= 8 ? 1 : Math.ceil(lastIndex / 6);
  return props.points
    .map((point, index) => ({ ...point, index, x: xAt(index) }))
    .filter(point => point.index === 0 || point.index === lastIndex || point.index % stride === 0);
});

function xAt(index: number): number {
  return props.points.length <= 1
    ? (plot.left + width - plot.right) / 2
    : plot.left + (index / (props.points.length - 1)) * (width - plot.left - plot.right);
}

function yAt(value: number): number {
  return height - plot.bottom - (value / yMaximum.value) * (height - plot.top - plot.bottom);
}

function compactNumber(value: number): string {
  if (value >= 1_000_000) return `${Number((value / 1_000_000).toPrecision(3))}M`;
  if (value >= 1_000) return `${Number((value / 1_000).toPrecision(3))}k`;
  return Number(value.toPrecision(3)).toLocaleString("en-US");
}

function formatAxisValue(value: number): string {
  if (value === 0) return "0";
  if (props.valueLabel.includes("时长")) {
    return value >= 1000 ? `${compactNumber(value / 1000)}s` : `${compactNumber(value)}ms`;
  }
  return compactNumber(value);
}

function formatEntityTick(id: number): string {
  return props.entityLabel === "Run" ? `#${id}` : `R${id}`;
}

const plottedPoints = computed(() =>
  props.points.map((point, index) => ({ ...point, x: xAt(index), y: point.value === null ? null : yAt(point.value) })),
);
const segments = computed(() => {
  const result: string[] = [];
  let current: string[] = [];
  for (const point of plottedPoints.value) {
    if (point.value === null || point.y === null) {
      if (current.length > 1) result.push(current.join(" "));
      current = [];
    } else {
      current.push(`${current.length === 0 ? "M" : "L"} ${point.x} ${point.y}`);
    }
  }
  if (current.length > 1) result.push(current.join(" "));
  return result;
});
const activePoint = computed(() => plottedPoints.value.find(point => point.id === activeId.value) ?? null);
</script>

<template>
  <article
    class="trend-card"
    :data-testid="testId ?? `trend-${title.toLowerCase().replaceAll(' ', '-')}`"
  >
    <header>
      <div><p>{{ entityLabel }} 趋势</p><h3>{{ title }}</h3></div>
      <span>{{ points.length }} 个 {{ entityLabel }} · 按时间顺序</span>
    </header>
    <p class="description">
      {{ description }} 缺失观测保留为空缺，不按 0 处理。
    </p>
    <div
      v-if="points.length === 0"
      class="empty"
    >
      暂无符合条件的 {{ entityLabel }}。
    </div>
    <div
      v-else
      class="chart-wrap"
    >
      <svg
        :viewBox="`0 0 ${width} ${height}`"
        role="img"
        :aria-label="title"
      >
        <g class="y-axis">
          <g
            v-for="tick in yTicks"
            :key="tick"
            class="y-axis-tick"
          >
            <line
              :x1="plot.left"
              :x2="width - plot.right"
              :y1="yAt(tick)"
              :y2="yAt(tick)"
              class="grid-line"
            />
            <text
              :x="plot.left - 8"
              :y="yAt(tick)"
              class="axis-label y-tick-label"
              text-anchor="end"
              dominant-baseline="middle"
            >{{ formatAxisValue(tick) }}</text>
          </g>
        </g>
        <line
          :x1="plot.left"
          :x2="plot.left"
          :y1="plot.top"
          :y2="height - plot.bottom"
          class="axis-line"
        />
        <line
          :x1="plot.left"
          :x2="width - plot.right"
          :y1="height - plot.bottom"
          :y2="height - plot.bottom"
          class="axis-line"
        />
        <path
          v-for="segment in segments"
          :key="segment"
          :d="segment"
          class="trend-line"
        />
        <circle
          v-for="point in plottedPoints.filter(item => item.value !== null)"
          :key="point.id"
          :cx="point.x"
          :cy="point.y!"
          r="4.5"
          tabindex="0"
          class="data-point"
          :data-point-id="point.id"
          :data-value="point.value"
          :aria-label="`${entityLabel} ${point.id}, ${valueLabel}: ${formatValue(point.value!)}`"
          @mouseenter="activeId = point.id"
          @mouseleave="activeId = null"
          @focus="activeId = point.id"
          @blur="activeId = null"
        />
        <text
          v-for="tick in xTicks"
          :key="tick.id"
          :x="tick.x"
          :y="height - 13"
          class="axis-label x-tick-label"
          text-anchor="middle"
        >{{ formatEntityTick(tick.id) }}</text>
      </svg>
      <div
        v-if="activePoint?.value !== null && activePoint?.value !== undefined"
        class="tooltip"
        :style="{ left: `${(activePoint.x / width) * 100}%`, top: `${(activePoint.y! / height) * 100}%` }"
        role="status"
      >
        <strong>{{ entityLabel }} #{{ activePoint.id }}</strong><span>{{ valueLabel }} · {{ formatValue(activePoint.value) }}</span>
      </div>
    </div>
    <ol class="sr-only">
      <li
        v-for="point in points"
        :key="point.id"
      >
        {{ entityLabel }} {{ point.id }}：{{ point.value === null ? "未捕获" : formatValue(point.value) }}
      </li>
    </ol>
  </article>
</template>

<style scoped>
.trend-card { position:relative; min-width:0; overflow:hidden; border:1px solid var(--strong-border); border-radius:var(--radius-lg); background:color-mix(in srgb,var(--surface) 94%,var(--accent)); box-shadow:inset 0 2px 0 color-mix(in srgb,var(--primary) 52%,transparent),0 16px 34px rgba(31,45,49,.05); }
.trend-card::after { position:absolute; right:-2rem; bottom:-3.5rem; width:9rem; height:9rem; border:1px solid color-mix(in srgb,var(--primary) 9%,transparent); border-radius:50%; box-shadow:0 0 0 1.6rem color-mix(in srgb,var(--primary) 3%,transparent); content:""; pointer-events:none; }
header { position:relative; z-index:1; display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding:1rem 1.1rem 0; }
header p, header span { margin:0; color:var(--muted-foreground); font-size:.75rem; }
header p { color:var(--primary-deep); font-weight:650; letter-spacing:.04em; }
h3 { margin:.18rem 0 0; font-size:1.1rem; font-weight:650; letter-spacing:-.018em; }
.description { margin:0; padding:.45rem 1.1rem 0; color:var(--muted-foreground); font-size:.8125rem; line-height:1.5; }
.chart-wrap { position:relative; z-index:1; padding:.45rem .55rem .75rem; }
svg { display:block; width:100%; height:13.125rem; overflow:visible; }
.grid-line { stroke:color-mix(in srgb,var(--primary) 18%,var(--border)); stroke-width:1; stroke-dasharray:3 5; }
.axis-line { stroke:color-mix(in srgb,var(--foreground) 24%,var(--border)); stroke-width:1; vector-effect:non-scaling-stroke; }
.axis-label { fill:var(--muted-foreground); font-family:var(--langley-font-mono); font-size:11px; font-variant-numeric:tabular-nums; }
.trend-line { fill:none; stroke:var(--primary); stroke-width:2; vector-effect:non-scaling-stroke; }
.data-point { fill:var(--surface); stroke:var(--primary-deep); stroke-width:2.2; cursor:crosshair; vector-effect:non-scaling-stroke; }
.data-point:hover, .data-point:focus { fill:var(--primary); stroke-width:3; outline:none; }
.tooltip { position:absolute; z-index:2; display:grid; gap:.15rem; min-width:8.5rem; padding:.5rem .6rem; transform:translate(-50%,calc(-100% - .5rem)); border:1px solid var(--strong-border); border-radius:var(--radius-sm); background:var(--foreground); color:var(--background); font-size:.75rem; line-height:1.4; font-variant-numeric:tabular-nums; pointer-events:none; }
.tooltip span { opacity:.78; }
.empty { padding:2.7rem 1.1rem; color:var(--muted-foreground); font-size:.8125rem; }
.sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; clip-path:inset(50%); }
</style>
