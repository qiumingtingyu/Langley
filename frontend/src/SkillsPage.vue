<script setup lang="ts">
import { BookMarked, FileText, RotateCw, ShieldCheck } from "lucide-vue-next";
import { onMounted, ref } from "vue";

import SkillInstructions from "@/components/SkillInstructions.vue";
import { Button } from "@/components/ui/button";
import type { SkillDetail, SkillSummary } from "@/types";

class SkillRequestError extends Error {
  constructor(readonly code: string | undefined) {
    super(code);
  }
}

const skills = ref<SkillSummary[]>([]);
const selectedName = ref<string | null>(null);
const detail = ref<SkillDetail | null>(null);
const inventoryLoading = ref(true);
const detailLoading = ref(false);
const inventoryError = ref<string | null>(null);
const detailError = ref<string | null>(null);
const detailErrorCode = ref<string | null>(null);
let inventoryRevision = 0;
let detailRevision = 0;

function errorCode(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return;
  const responseDetail = payload.detail;
  if (typeof responseDetail !== "object" || responseDetail === null || !("code" in responseDetail)) return;
  return typeof responseDetail.code === "string" ? responseDetail.code : undefined;
}

function errorMessage(error: unknown): string {
  const code = error instanceof SkillRequestError ? error.code : undefined;
  if (code === "SKILL_CATALOG_UNAVAILABLE") return "技能目录暂时不可用，请稍后重试。";
  if (code === "SKILL_NOT_FOUND") return "这个内置技能已不存在，请刷新后重试。";
  if (code === "SKILL_CONTENT_UNAVAILABLE") return "技能内容暂时不可用，请稍后重试。";
  return "技能请求未完成，请稍后重试。";
}

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new SkillRequestError(undefined);
  }
  if (!response.ok) throw new SkillRequestError(errorCode(payload));
  return payload as T;
}

async function loadInventory(): Promise<void> {
  const revision = ++inventoryRevision;
  inventoryLoading.value = true;
  inventoryError.value = null;
  try {
    const nextSkills = await requestJson<SkillSummary[]>("/api/skills");
    if (revision !== inventoryRevision) return;
    skills.value = nextSkills;
    inventoryLoading.value = false;
    const nextName = nextSkills.some((skill) => skill.name === selectedName.value)
      ? selectedName.value
      : nextSkills[0]?.name ?? null;
    if (nextName === null) {
      selectedName.value = null;
      detail.value = null;
      detailError.value = null;
      detailErrorCode.value = null;
      detailLoading.value = false;
      detailRevision += 1;
      return;
    }
    void selectSkill(nextName);
  } catch (error) {
    if (revision !== inventoryRevision) return;
    skills.value = [];
    selectedName.value = null;
    detail.value = null;
    detailRevision += 1;
    inventoryError.value = errorMessage(error);
  } finally {
    if (revision === inventoryRevision) inventoryLoading.value = false;
  }
}

async function selectSkill(name: string): Promise<void> {
  const revision = ++detailRevision;
  selectedName.value = name;
  detail.value = null;
  detailError.value = null;
  detailErrorCode.value = null;
  detailLoading.value = true;
  try {
    const nextDetail = await requestJson<SkillDetail>(`/api/skills/${encodeURIComponent(name)}`);
    if (revision !== detailRevision || selectedName.value !== name) return;
    detail.value = nextDetail;
  } catch (error) {
    if (revision !== detailRevision || selectedName.value !== name) return;
    detailError.value = errorMessage(error);
    detailErrorCode.value = error instanceof SkillRequestError ? error.code ?? null : null;
  } finally {
    if (revision === detailRevision && selectedName.value === name) detailLoading.value = false;
  }
}

function retryDetail(): void {
  if (selectedName.value !== null) void selectSkill(selectedName.value);
}

function recoverDetail(): void {
  if (detailErrorCode.value !== "SKILL_NOT_FOUND") {
    retryDetail();
    return;
  }
  detailRevision += 1;
  detail.value = null;
  detailError.value = null;
  detailErrorCode.value = null;
  detailLoading.value = false;
  void loadInventory();
}

function formatByteSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kibibytes = bytes / 1024;
  return `${kibibytes >= 10 ? kibibytes.toFixed(0) : kibibytes.toFixed(1)} KiB`;
}

onMounted(() => void loadInventory());
</script>

<template>
  <div class="mx-auto flex min-h-full w-full max-w-[82rem] flex-col px-4 py-7 sm:px-6 lg:h-full lg:px-8 lg:py-9">
    <header class="shrink-0 border-b border-border pb-6">
      <div class="flex items-start gap-3">
        <span class="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md border border-strong-border bg-surface text-primary-deep">
          <BookMarked
            :size="18"
            :stroke-width="1.6"
            aria-hidden="true"
          />
        </span>
        <div>
          <h1 class="text-2xl font-semibold tracking-[-0.025em] text-foreground">
            技能
          </h1>
          <p class="mt-1 text-sm leading-6 text-muted-foreground">
            查看 Agent 可按需加载的内置任务方法与流程。
          </p>
        </div>
      </div>
    </header>

    <div class="mt-6 grid min-h-0 flex-1 gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]">
      <aside class="min-h-0 rounded-lg border border-border bg-sidebar/55 p-3 lg:overflow-y-auto">
        <div class="flex items-center justify-between gap-3 px-2 pb-3">
          <h2 class="text-xs font-semibold tracking-[0.08em] text-muted-foreground uppercase">
            内置技能
          </h2>
          <span
            v-if="!inventoryLoading && inventoryError === null"
            class="font-mono text-[11px] tabular-nums text-muted-light"
          >{{ skills.length }}</span>
        </div>

        <p
          v-if="inventoryLoading"
          role="status"
          class="px-2 py-6 text-sm text-muted-foreground"
        >
          正在读取技能目录…
        </p>
        <div
          v-else-if="inventoryError !== null"
          class="flex flex-col items-start gap-3 px-2 py-5"
        >
          <p
            role="alert"
            class="text-sm leading-6 text-danger-foreground"
          >
            {{ inventoryError }}
          </p>
          <Button
            variant="outline"
            size="small"
            @click="loadInventory"
          >
            <RotateCw data-icon="inline-start" />
            重试
          </Button>
        </div>
        <p
          v-else-if="skills.length === 0"
          class="px-2 py-6 text-sm leading-6 text-muted-foreground"
        >
          当前没有内置 Skill。
        </p>
        <div
          v-else
          class="flex flex-col gap-1"
        >
          <button
            v-for="skill in skills"
            :key="skill.name"
            type="button"
            class="group rounded-md px-3 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
            :class="selectedName === skill.name ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-body hover:bg-subtle'"
            :aria-pressed="selectedName === skill.name"
            @click="selectSkill(skill.name)"
          >
            <span class="block truncate font-mono text-[13px] font-semibold">{{ skill.name }}</span>
            <span class="mt-1.5 line-clamp-2 block text-xs leading-5 text-muted-foreground">{{ skill.description }}</span>
            <span class="mt-2 inline-flex rounded-sm border border-strong-border bg-accent px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-accent-foreground">内置</span>
          </button>
        </div>
      </aside>

      <article class="min-h-[28rem] min-w-0 rounded-lg border border-border bg-surface lg:overflow-y-auto">
        <div
          v-if="selectedName === null && !inventoryLoading && inventoryError === null && skills.length > 0"
          class="p-6 text-sm text-muted-foreground"
        >
          选择一个技能查看详情。
        </div>
        <div
          v-else-if="detailLoading"
          role="status"
          class="p-6 text-sm text-muted-foreground sm:p-8"
        >
          正在读取技能内容…
        </div>
        <div
          v-else-if="detailError !== null"
          class="flex flex-col items-start gap-4 p-6 sm:p-8"
        >
          <p
            role="alert"
            class="text-sm leading-6 text-danger-foreground"
          >
            {{ detailError }}
          </p>
          <Button
            variant="outline"
            size="small"
            @click="recoverDetail"
          >
            <RotateCw data-icon="inline-start" />
            {{ detailErrorCode === "SKILL_NOT_FOUND" ? "刷新技能目录" : "重试内容" }}
          </Button>
        </div>
        <div
          v-else-if="detail !== null"
          class="p-6 sm:p-8 lg:p-10"
        >
          <div class="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-7">
            <div class="min-w-0">
              <div class="flex flex-wrap items-center gap-2.5">
                <h2 class="break-words font-mono text-xl font-semibold tracking-[-0.02em] text-foreground sm:text-2xl">
                  {{ detail.name }}
                </h2>
                <span class="inline-flex rounded-sm border border-strong-border bg-accent px-2 py-0.5 text-[11px] font-medium tracking-wide text-accent-foreground">内置</span>
              </div>
              <p class="mt-3 max-w-3xl text-sm leading-6 text-body">
                {{ detail.description }}
              </p>
            </div>
          </div>

          <section class="border-b border-border py-7">
            <div class="flex items-center gap-2 text-primary-deep">
              <ShieldCheck
                :size="17"
                :stroke-width="1.7"
                aria-hidden="true"
              />
              <h3 class="text-sm font-semibold text-foreground">
                运行语义
              </h3>
            </div>
            <p class="mt-3 max-w-3xl text-sm leading-7 text-muted-foreground">
              Skill 为 Agent 提供可复用的方法和流程，不会授予新的工具权限。Agent 在任务需要时显式加载 Skill；一次 Run 最多激活一个 Skill。
            </p>
          </section>

          <section class="border-b border-border py-7">
            <div class="mb-5 flex items-center gap-2">
              <FileText
                :size="17"
                :stroke-width="1.7"
                aria-hidden="true"
                class="text-primary-deep"
              />
              <h3 class="text-sm font-semibold text-foreground">
                技能说明
              </h3>
            </div>
            <SkillInstructions :content="detail.instructions" />
          </section>

          <section class="pt-7">
            <h3 class="text-sm font-semibold text-foreground">
              参考资料
            </h3>
            <p
              v-if="detail.resources.length === 0"
              class="mt-3 text-sm leading-6 text-muted-foreground"
            >
              当前 Skill 没有附带 references。
            </p>
            <ul
              v-else
              class="mt-4 flex flex-col gap-2"
            >
              <li
                v-for="resource in detail.resources"
                :key="resource.path"
                class="flex min-w-0 items-baseline justify-between gap-4 rounded-md border border-border bg-subtle px-3.5 py-3"
              >
                <span class="min-w-0 break-all font-mono text-xs leading-5 text-body">{{ resource.path }}</span>
                <span class="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{{ formatByteSize(resource.byte_size) }}</span>
              </li>
            </ul>
          </section>
        </div>
      </article>
    </div>
  </div>
</template>
