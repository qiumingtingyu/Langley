<script setup lang="ts">
import { DialogContent, DialogDescription, DialogOverlay, DialogPortal, DialogRoot, DialogTitle } from "reka-ui";
import { computed, onMounted, ref } from "vue";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
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
const editorOpen = ref(false);
const editorError = ref("");
const forgetOpen = ref(false);
const forgetError = ref("");
const forgetTarget = ref<Memory | null>(null);
const pageTitle = ref<HTMLElement | null>(null);
let editorTrigger: HTMLElement | null = null;
let forgetTrigger: HTMLElement | null = null;
const source = ref<Source | null>(null);
const sourceFor = ref<number | null>(null);
const sourceLoading = ref(false);
const syncResult = ref<string | null>(null);
let loadRevision = 0;
let sourceRevision = 0;

const autoMemoryEnabled = computed(() => memoryStatus.value?.auto_memory_enabled ?? null);
const policyReady = computed(() => memoryStatus.value?.policy_status === "READY");

function errorCode(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return;
  const detail = payload.detail;
  if (typeof detail !== "object" || detail === null || !("code" in detail)) return;
  return typeof detail.code === "string" ? detail.code : undefined;
}

function readableError(error: unknown): string {
  return error instanceof MemoryRequestError ? error.message : "操作失败，请稍后重试。";
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
  return status === "READY" ? "整理服务可用" : "自动整理暂不可用";
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
    if (editing.value !== null && !nextMemories.some((memory) => memory.id === editing.value)) {
      resetForm();
      editorOpen.value = false;
    }
  } catch (error) {
    if (revision === loadRevision) emit("notice", readableError(error));
  } finally {
    if (revision === loadRevision) loading.value = false;
  }
}

function absoluteValidUntil(): string | null {
  return validUntil.value ? new Date(validUntil.value).toISOString() : null;
}

function resetForm(): void {
  editorError.value = "";
  content.value = "";
  validUntil.value = "";
  editing.value = null;
}

async function refreshAfterNotFound(error: unknown): Promise<void> {
  if (error instanceof MemoryRequestError && error.code === "MEMORY_NOT_FOUND") await load();
}

async function save(): Promise<void> {
  if (savingMemory.value) return;
  editorError.value = "";
  if (!content.value.trim()) {
    editorError.value = "输入内容无效，请检查后重试。";
    emit("notice", editorError.value);
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
    editorOpen.value = false;
    await load();
    emit("notice", "长期记忆已更新。");
  } catch (error) {
    await load();
    editorError.value = readableError(error);
    emit("notice", readableError(error));
  } finally {
    savingMemory.value = false;
  }
}

async function toggle(): Promise<void> {
  if (updatingAutoMemory.value || memoryStatus.value === null) return;
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
    emit("notice", readableError(error));
  } finally {
    updatingAutoMemory.value = false;
  }
}

async function sync(): Promise<void> {
  if (syncingMemory.value || !policyReady.value) return;
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
    emit("notice", readableError(error));
  } finally {
    syncingMemory.value = false;
  }
}

function localDatetimeInput(value: string): string {
  const date = new Date(value);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 19);
}

function addMemory(event: Event): void {
  if (savingMemory.value) return;
  editorTrigger = event.currentTarget as HTMLElement;
  resetForm();
  editorOpen.value = true;
}

function setEditorOpen(open: boolean): void {
  if (savingMemory.value) return;
  editorOpen.value = open;
  if (!open) resetForm();
}

function restoreEditorFocus(event: Event): void {
  event.preventDefault();
  (editorTrigger?.isConnected ? editorTrigger : pageTitle.value)?.focus();
}

function restoreForgetFocus(event: Event): void {
  event.preventDefault();
  (forgetTrigger?.isConnected ? forgetTrigger : pageTitle.value)?.focus();
}

function edit(memory: Memory, event: Event): void {
  if (savingMemory.value) return;
  editorError.value = "";
  editorTrigger = event.currentTarget as HTMLElement;
  editing.value = memory.id;
  content.value = memory.content;
  validUntil.value = memory.valid_until ? localDatetimeInput(memory.valid_until) : "";
  editorOpen.value = true;
}

function askForget(memory: Memory, event: Event): void {
  if (savingMemory.value) return;
  forgetError.value = "";
  forgetTrigger = event.currentTarget as HTMLElement;
  forgetTarget.value = memory;
  forgetOpen.value = true;
}

function setForgetOpen(open: boolean): void {
  if (savingMemory.value) return;
  forgetOpen.value = open;
  if (!open) forgetTarget.value = null;
}

async function forget(): Promise<void> {
  if (savingMemory.value || forgetTarget.value === null) return;
  forgetError.value = "";
  const memory = forgetTarget.value;
  savingMemory.value = true;
  try {
    await request(`/api/memories/${memory.id}`, { method: "DELETE" });
    await load();
    forgetOpen.value = false;
    forgetTarget.value = null;
  } catch (error) {
    await load();
    if (error instanceof MemoryRequestError && error.code === "MEMORY_NOT_FOUND") {
      forgetOpen.value = false;
      forgetTarget.value = null;
    }
    forgetError.value = readableError(error);
    emit("notice", readableError(error));
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
      emit("notice", readableError(error));
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
  <section class="min-h-0 w-full bg-workspace px-5 py-7 text-sm sm:px-8 lg:px-10">
    <div class="mx-auto w-full max-w-5xl">
      <header class="border-b border-border pb-5">
        <div class="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1
              ref="pageTitle"
              tabindex="-1"
              class="text-[28px] font-semibold leading-[1.25] text-foreground"
            >
              长期记忆
            </h1>
            <p class="mt-2 text-[13px] font-normal text-muted-foreground">
              {{ memories.length }} 条内容 · {{ autoMemoryEnabled === null ? "正在读取状态…" : autoMemoryEnabled ? "自动整理已开启" : "自动整理已关闭" }}
            </p>
          </div>
          <Button
            type="button"
            :disabled="savingMemory"
            @click="addMemory"
          >
            添加记忆
          </Button>
        </div>
        <p class="mt-4 text-[15px] leading-[1.7] text-body">
          这些信息会在未来对话中帮助 Langley 更了解你的长期偏好和背景。
        </p>
      </header>

      <section
        aria-label="自动整理"
        class="border-b border-border py-4"
      >
        <div class="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          <div class="mr-auto">
            <h2 class="text-[17px] font-semibold text-foreground">
              自动整理
            </h2>
            <p class="mt-1 text-[13px] text-muted-foreground">
              从对话中保留长期有用的信息。
            </p>
          </div>
          <p class="rounded-sm border border-border bg-subtle px-2 py-0.5 text-[13px] text-body">
            {{ autoMemoryEnabled === null ? "正在读取状态…" : autoMemoryEnabled ? "已开启" : "已关闭" }}
          </p>
          <p
            class="text-[13px]"
            :class="memoryStatus ? policyReady ? 'text-success-foreground' : 'text-warning-foreground' : 'text-muted-foreground'"
          >
            {{ policyStatusText(memoryStatus?.policy_status) }}
          </p>
          <Button
            type="button"
            variant="outline"
            size="small"
            :disabled="updatingAutoMemory || memoryStatus === null"
            @click="toggle"
          >
            {{ updatingAutoMemory ? "正在更新…" : memoryStatus === null ? "正在读取状态…" : autoMemoryEnabled ? "关闭自动整理" : "开启自动整理" }}
          </Button>
        </div>
        <details class="mt-3 text-[13px] leading-6 text-muted-foreground">
          <summary class="cursor-pointer focus-visible:outline focus-visible:outline-ring">
            整理设置说明
          </summary>
          <p class="mt-2">
            系统会从普通对话中整理长期有用的信息。关闭后不再自动保存普通对话中的信息；你明确提出的记住、修改或忘记请求仍可被处理。
          </p>
          <p v-if="memoryStatus && memoryStatus.pending_evidence_count > 0 && oldestPendingText(memoryStatus.oldest_pending_created_at)">
            {{ oldestPendingText(memoryStatus.oldest_pending_created_at) }}
          </p>
        </details>
        <div
          v-if="memoryStatus && memoryStatus.pending_evidence_count > 0"
          class="mt-3 flex flex-wrap items-center gap-3 border-l-2 border-primary pl-3 text-sm"
        >
          <p
            role="status"
            class="text-body"
          >
            还有 {{ memoryStatus.pending_evidence_count }} 条内容待整理
          </p>
          <Button
            type="button"
            variant="outline"
            size="small"
            :disabled="syncingMemory || !policyReady"
            @click="sync"
          >
            {{ syncingMemory ? "正在整理…" : "立即整理" }}
          </Button>
          <p
            v-if="!policyReady"
            class="text-warning-foreground"
          >
            自动整理当前不可用，待整理内容会保留在这里。
          </p>
        </div>
        <p
          v-if="syncResult"
          role="status"
          class="mt-3 text-sm text-body"
        >
          {{ syncResult }}
        </p>
      </section>

      <section
        aria-label="记忆内容"
        class="py-2"
      >
        <p
          v-if="loading"
          class="py-6 text-sm text-muted-foreground"
        >
          正在加载…
        </p>
        <div
          v-else-if="memories.length === 0"
          class="py-10 text-sm leading-6 text-muted-foreground"
        >
          <p class="text-base font-medium text-foreground">
            还没有长期记忆
          </p>
          <p class="mt-2 text-sm leading-[1.7]">
            {{ autoMemoryEnabled === null ? "你可以手动添加长期信息。" : autoMemoryEnabled ? "Langley 会从对话中整理长期有用的信息，也可以手动添加。" : "自动整理已关闭，你仍可手动添加或明确要求 Langley 记住。" }}
          </p>
        </div>
        <div
          v-else
          class="divide-y divide-border"
        >
          <article
            v-for="memory in memories"
            :key="memory.id"
            class="py-6"
          >
            <p class="break-words whitespace-pre-wrap text-[15.5px] font-normal leading-[1.75] text-foreground">
              {{ memory.content }}
            </p>
            <p class="mt-3 text-[13px] font-normal text-muted-foreground">
              {{ memory.valid_until ? `有效至：${new Date(memory.valid_until).toLocaleString()}` : "长期有效" }}
            </p>
            <div class="mt-2 flex flex-wrap gap-x-2 gap-y-1">
              <Button
                type="button"
                size="small"
                variant="outline"
                @click="showSource(memory)"
              >
                查看来源
              </Button>
              <Button
                type="button"
                size="small"
                variant="outline"
                :disabled="savingMemory"
                @click="edit(memory, $event)"
              >
                修改
              </Button>
              <Button
                type="button"
                size="small"
                variant="outline"
                class="text-danger-foreground"
                :disabled="savingMemory"
                @click="askForget(memory, $event)"
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
                <div class="mt-3 flex flex-col gap-2">
                  <p
                    v-for="message in source.context_messages"
                    :key="message.id"
                    class="break-words whitespace-pre-wrap border-l-2 pl-3 leading-6"
                    :class="message.id === memory.source_message_id ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'"
                  >
                    <span
                      v-if="message.id === memory.source_message_id"
                      class="mb-1 block text-xs font-normal text-muted-foreground"
                    >相关对话内容</span>
                    {{ message.content }}
                  </p>
                </div>
              </template>
            </section>
          </article>
        </div>
      </section>
    </div>

    <Sheet
      :open="editorOpen"
      @update:open="setEditorOpen"
    >
      <SheetContent
        side="right"
        class="w-full overflow-y-auto bg-surface p-5 shadow-none sm:max-w-lg sm:p-6"
        @close-auto-focus="restoreEditorFocus"
      >
        <SheetHeader class="p-0 pr-8">
          <SheetTitle class="text-xl font-semibold">
            {{ editing === null ? "添加长期信息" : "修改长期信息" }}
          </SheetTitle>
          <SheetDescription>只添加会在未来对话中持续有帮助的信息。</SheetDescription>
        </SheetHeader>
        <form
          class="mt-3 flex flex-col gap-4"
          @submit.prevent="save"
        >
          <p
            v-if="editorError"
            role="alert"
            class="text-sm text-danger-foreground"
          >
            {{ editorError }}
          </p>
          <label class="block text-sm font-medium text-body">长期信息<textarea
            v-model="content"
            class="memory-content-input mt-2 min-h-40 w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
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
              {{ savingMemory ? "正在保存…" : "保存" }}
            </Button>
            <Button
              type="button"
              variant="ghost"
              :disabled="savingMemory"
              @click="setEditorOpen(false)"
            >
              取消
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>

    <DialogRoot
      :open="forgetOpen"
      @update:open="setForgetOpen"
    >
      <DialogPortal>
        <DialogOverlay class="fixed inset-0 z-40 bg-foreground/20" />
        <DialogContent
          class="fixed left-1/2 top-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-md border border-strong-border bg-surface p-5 text-foreground outline-none sm:p-6"
          @close-auto-focus="restoreForgetFocus"
        >
          <DialogTitle class="text-xl font-semibold">
            忘记这条长期记忆？
          </DialogTitle>
          <DialogDescription class="mt-3 text-sm leading-[1.6] text-muted-foreground">
            这条信息将不再用于未来对话。
          </DialogDescription>
          <p
            v-if="forgetError"
            role="alert"
            class="mt-3 text-sm text-danger-foreground"
          >
            {{ forgetError }}
          </p>
          <div class="mt-5 flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              :disabled="savingMemory"
              @click="setForgetOpen(false)"
            >
              取消
            </Button>
            <Button
              type="button"
              variant="outline"
              class="border-danger-border text-danger-foreground hover:bg-danger-surface"
              :disabled="savingMemory"
              @click="forget"
            >
              {{ savingMemory ? "正在忘记…" : "忘记" }}
            </Button>
          </div>
        </DialogContent>
      </DialogPortal>
    </DialogRoot>
  </section>
</template>

<style scoped>
/* Native controls inherit fonts after Tailwind's layers; keep these local. */
button[data-slot="button"] {
  font-size: 14px;
}

.memory-content-input {
  font-size: 15px;
  font-weight: 400;
  line-height: 1.7;
}

input {
  font-weight: 400;
}
</style>
