<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { Button } from "@/components/ui/button";
import type { Memory, MemoryOperationalStatus, MemoryPolicyStatus, MemorySyncResult } from "@/types";

type Source =
  | { kind: "direct" }
  | {
    kind: "conversation";
    conversation_title: string | null;
    conversation_deleted: boolean;
    context_messages: Array<{ id: number; role: "USER" | "ASSISTANT"; content: string }>;
  };

class MemoryRequestError extends Error {
  constructor(readonly code: string | undefined) {
    super(errorMessage(code));
  }
}

const emit = defineEmits<{ notice: [message: string]; }>();
const memories = ref<Memory[]>([]);
const memoryStatus = ref<MemoryOperationalStatus | null>(null);
const loading = ref(false);
const savingMemory = ref(false);
const updatingAutoMemory = ref(false);
const syncingMemory = ref(false);
const content = ref("");
const validUntil = ref("");
const editing = ref<number | null>(null);
const source = ref<Source | null>(null);
const sourceFor = ref<number | null>(null);
const sourceLoading = ref(false);
const syncResult = ref<string | null>(null);
let loadRevision = 0;
let sourceRevision = 0;

const autoMemoryEnabled = computed(() => memoryStatus.value?.auto_memory_enabled ?? null);
const policyReady = computed(() => memoryStatus.value?.policy_status === "READY");

function errorCode(payload: unknown): string | undefined {
  return typeof payload === "object" && payload !== null && "detail" in payload
    ? (payload.detail as { code?: string }).code
    : undefined;
}

function errorMessage(code: string | undefined): string {
  if (code === "MEMORY_NOT_FOUND") return "这条长期记忆已不存在。";
  if (code === "MEMORY_SYNC_UNAVAILABLE") return "自动整理暂时不可用，请稍后重试。";
  if (code === "MEMORY_SYNC_BLOCKED") return "当前记忆状态暂时无法继续自动整理。";
  if (code === "MEMORY_SYNC_INCOMPLETE") return "还有内容尚未整理完成，请继续整理后再试。";
  if (code === "MEMORY_CAPACITY_REACHED") return "当前长期记忆已达到可处理容量，请先整理或删除部分内容。";
  if (code === "MEMORY_CAPACITY_UNAVAILABLE") return "当前无法确认新增记忆容量，请稍后重试。";
  if (code === "VALIDATION_ERROR") return "输入内容无效，请检查后重试。";
  return "操作失败，请稍后重试。";
}

function policyStatusText(status: MemoryPolicyStatus | undefined): string {
  if (status === undefined) return "正在读取状态…";
  return status === "READY" ? "自动整理可用" : "自动整理暂不可用";
}

async function request(path: string, init?: Parameters<typeof window.fetch>[1]): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) throw new MemoryRequestError(errorCode(await response.json()));
  return response;
}

async function load(): Promise<void> {
  const revision = ++loadRevision;
  loading.value = true;
  try {
    const [statusResponse, memoriesResponse] = await Promise.all([request("/api/memory-status"), request("/api/memories")]);
    const [nextStatus, nextMemories] = await Promise.all([
      statusResponse.json() as Promise<MemoryOperationalStatus>,
      memoriesResponse.json() as Promise<Memory[]>,
    ]);
    if (revision !== loadRevision) return;
    memoryStatus.value = nextStatus;
    memories.value = nextMemories;
    if (editing.value !== null && !nextMemories.some((memory) => memory.id === editing.value)) resetForm();
  } catch (error) {
    if (revision === loadRevision) emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally {
    if (revision === loadRevision) loading.value = false;
  }
}

function absoluteValidUntil(): string | null {
  return validUntil.value ? new Date(validUntil.value).toISOString() : null;
}

function resetForm(): void {
  content.value = "";
  validUntil.value = "";
  editing.value = null;
}

async function refreshAfterNotFound(error: unknown): Promise<void> {
  if (error instanceof MemoryRequestError && error.code === "MEMORY_NOT_FOUND") await load();
}

async function save(): Promise<void> {
  if (!content.value.trim()) {
    emit("notice", "输入内容无效，请检查后重试。");
    return;
  }
  savingMemory.value = true;
  try {
    const path = editing.value === null ? "/api/memories" : `/api/memories/${editing.value}`;
    await request(path, {
      method: editing.value === null ? "POST" : "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: content.value.trim(), valid_until: absoluteValidUntil() }),
    });
    resetForm();
    await load();
    emit("notice", "长期记忆已更新。");
  } catch (error) {
    await load();
    emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally {
    savingMemory.value = false;
  }
}

async function toggle(): Promise<void> {
  if (memoryStatus.value === null) return;
  updatingAutoMemory.value = true;
  try {
    await request("/api/memory-settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_memory_enabled: !memoryStatus.value.auto_memory_enabled }),
    });
    await load();
  } catch (error) {
    await load();
    emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally {
    updatingAutoMemory.value = false;
  }
}

async function sync(): Promise<void> {
  if (!policyReady.value) return;
  syncingMemory.value = true;
  syncResult.value = null;
  try {
    const result = await (await request("/api/memory-sync", { method: "POST" })).json() as MemorySyncResult;
    syncResult.value = result.complete && result.remaining_count === 0
      ? "已整理完成。"
      : `本次整理了 ${result.processed_count} 条，完成时还有 ${result.remaining_count} 条待整理。`;
    await load();
  } catch (error) {
    await load();
    emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally {
    syncingMemory.value = false;
  }
}

function localDatetimeInput(value: string): string {
  const date = new Date(value);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 19);
}

function edit(memory: Memory): void {
  editing.value = memory.id;
  content.value = memory.content;
  validUntil.value = memory.valid_until ? localDatetimeInput(memory.valid_until) : "";
}

async function forget(memory: Memory): Promise<void> {
  if (!window.confirm("确定忘记这条长期记忆吗？")) return;
  savingMemory.value = true;
  try {
    await request(`/api/memories/${memory.id}`, { method: "DELETE" });
    await load();
  } catch (error) {
    await load();
    emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally {
    savingMemory.value = false;
  }
}

async function showSource(memory: Memory): Promise<void> {
  const revision = ++sourceRevision;
  sourceFor.value = memory.id;
  source.value = null;
  sourceLoading.value = true;
  try {
    const next = await (await request(`/api/memories/${memory.id}/source`)).json() as Source;
    if (revision === sourceRevision && sourceFor.value === memory.id) source.value = next;
  } catch (error) {
    if (revision === sourceRevision && sourceFor.value === memory.id) {
      await refreshAfterNotFound(error);
      emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
    }
  } finally {
    if (revision === sourceRevision && sourceFor.value === memory.id) sourceLoading.value = false;
  }
}

function oldestPendingText(value: string | null): string | null {
  return value === null ? null : `最早待处理：${new Date(value).toLocaleString()}`;
}

defineExpose({ load });
onMounted(() => void load());
</script>

<template>
  <section class="min-h-0 w-full bg-workspace px-5 py-7 sm:px-8 lg:px-10">
    <div class="mx-auto w-full max-w-4xl">
      <header class="border-b border-border pb-6">
        <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
          LONG-TERM MEMORY
        </p>
        <h1 class="mt-1 text-xl font-semibold tracking-[-0.02em] text-foreground">
          长期记忆
        </h1>
        <p class="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
          Langley 只保留对未来对话有帮助的个人信息，你可以随时查看、修改或忘记。
        </p>

        <section class="mt-5 border-l-2 border-primary bg-subtle px-4 py-4">
          <div class="flex flex-wrap items-start justify-between gap-4">
            <div class="min-w-0">
              <h2 class="text-sm font-semibold text-foreground">
                自动整理对话中的长期信息
              </h2>
              <p class="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">
                开启后，Langley 会从普通对话中整理对未来有帮助的信息。关闭后，不再自动保存普通对话中的信息；你明确提出的记住、修改或忘记请求仍可被处理。
              </p>
            </div>
            <Button
              type="button"
              :disabled="updatingAutoMemory || memoryStatus === null"
              @click="toggle"
            >
              {{ updatingAutoMemory ? "正在更新…" : memoryStatus === null ? "正在读取状态…" : autoMemoryEnabled ? "关闭自动整理" : "开启自动整理" }}
            </Button>
          </div>
          <div class="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-border pt-3 text-sm text-body">
            <p>状态：{{ policyStatusText(memoryStatus?.policy_status) }}</p>
            <template v-if="memoryStatus && memoryStatus.pending_evidence_count > 0">
              <p>待整理内容：{{ memoryStatus.pending_evidence_count }} 条</p>
              <p
                v-if="oldestPendingText(memoryStatus.oldest_pending_created_at)"
                class="text-muted-foreground"
              >
                {{ oldestPendingText(memoryStatus.oldest_pending_created_at) }}
              </p>
              <Button
                type="button"
                :disabled="syncingMemory || !policyReady"
                @click="sync"
              >
                {{ syncingMemory ? "正在整理…" : "立即整理" }}
              </Button>
            </template>
            <p
              v-else-if="memoryStatus"
              class="text-muted-foreground"
            >
              当前已整理
            </p>
          </div>
          <p
            v-if="memoryStatus && memoryStatus.pending_evidence_count > 0 && !policyReady"
            class="mt-3 text-sm text-muted-foreground"
          >
            自动整理当前不可用，待整理内容会保留在这里。
          </p>
          <p
            v-if="syncResult"
            class="mt-3 text-sm text-body"
          >
            {{ syncResult }}
          </p>
        </section>
      </header>

      <div class="grid gap-8 py-7 lg:grid-cols-[minmax(0,1fr)_minmax(15rem,0.6fr)]">
        <section class="min-w-0">
          <div class="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
            <div>
              <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
                CURRENT MEMORIES
              </p>
              <h2 class="mt-1 text-base font-semibold text-foreground">
                当前记住的内容
              </h2>
            </div>
            <p
              v-if="!loading"
              class="font-mono text-xs tabular-nums text-muted-light"
            >
              {{ memories.length }} 条
            </p>
          </div>

          <p
            v-if="loading"
            class="py-6 text-sm text-muted-foreground"
          >
            正在加载…
          </p>
          <div
            v-else-if="memories.length === 0"
            class="border-b border-dashed border-strong-border py-10 text-sm leading-6 text-muted-foreground"
          >
            <p>还没有长期记忆。</p>
            <p class="mt-1">
              {{ memoryStatus === null ? "正在读取自动整理状态。你仍可以手动添加长期信息。" : autoMemoryEnabled ? "Langley 会在对话中整理对未来有帮助的信息，你也可以手动添加。" : "自动整理当前已关闭，你仍可以手动添加，或在对话中明确要求 Langley 记住某件事。" }}
            </p>
          </div>
          <div
            v-else
            class="divide-y divide-border"
          >
            <article
              v-for="memory in memories"
              :key="memory.id"
              class="py-5"
            >
              <p class="break-words whitespace-pre-wrap text-sm leading-6 text-foreground">
                {{ memory.content }}
              </p>
              <p class="mt-2 text-xs text-muted-foreground">
                {{ memory.valid_until ? `有效至：${new Date(memory.valid_until).toLocaleString()}` : "长期有效" }}
              </p>
              <div class="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                <Button
                  size="small"
                  variant="ghost"
                  @click="showSource(memory)"
                >
                  查看来源
                </Button>
                <Button
                  size="small"
                  variant="ghost"
                  :disabled="savingMemory"
                  @click="edit(memory)"
                >
                  修改
                </Button>
                <Button
                  size="small"
                  variant="ghost"
                  :disabled="savingMemory"
                  @click="forget(memory)"
                >
                  忘记
                </Button>
              </div>
              <section
                v-if="sourceFor === memory.id && (sourceLoading || source)"
                class="mt-4 border-l-2 border-border bg-subtle px-3 py-3 text-sm text-body"
              >
                <p
                  v-if="sourceLoading"
                  class="text-muted-foreground"
                >
                  正在读取来源…
                </p>
                <template v-else-if="source?.kind === 'direct'">
                  <p>这条记忆由你直接添加或修改。</p>
                </template>
                <template v-else-if="source?.kind === 'conversation'">
                  <p class="text-muted-foreground">
                    {{ source.conversation_deleted ? "原会话已删除。" : `来自：${source.conversation_title ?? "未命名会话"}` }}
                  </p>
                  <div class="mt-3 space-y-2">
                    <p
                      v-for="message in source.context_messages"
                      :key="message.id"
                      class="break-words whitespace-pre-wrap border-l-2 pl-3 leading-6"
                      :class="message.id === memory.source_message_id ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'"
                    >
                      <span
                        v-if="message.id === memory.source_message_id"
                        class="mb-1 block font-mono text-[9px] font-medium tracking-[0.12em] text-muted-light"
                      >相关对话内容</span>
                      {{ message.content }}
                    </p>
                  </div>
                </template>
              </section>
            </article>
          </div>
        </section>

        <section class="min-w-0 border-t border-border pt-5 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
          <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
            MEMORY EDITOR
          </p>
          <h2 class="mt-1 text-base font-semibold text-foreground">
            {{ editing === null ? "添加长期信息" : "修改长期信息" }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-muted-foreground">
            只添加会在未来对话中持续有帮助的信息。
          </p>
          <form
            class="mt-5 space-y-4"
            @submit.prevent="save"
          >
            <label class="block text-sm font-medium text-body">长期信息<textarea
              v-model="content"
              class="mt-2 min-h-28 w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-sm leading-6 text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
              maxlength="1000"
              :disabled="savingMemory"
            /></label>
            <label class="block text-sm font-medium text-body">有效期（可选）<input
              v-model="validUntil"
              type="datetime-local"
              step="1"
              class="mt-2 block w-full max-w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
              :disabled="savingMemory"
            ></label>
            <p class="text-xs text-muted-foreground">
              留空表示长期有效。
            </p>
            <div class="flex flex-wrap gap-2">
              <Button
                type="submit"
                :disabled="savingMemory"
              >
                {{ savingMemory ? "正在保存…" : editing === null ? "添加长期信息" : "保存修改" }}
              </Button>
              <Button
                v-if="editing !== null"
                type="button"
                variant="ghost"
                @click="resetForm"
              >
                取消修改
              </Button>
            </div>
          </form>
        </section>
      </div>
    </div>
  </section>
</template>
