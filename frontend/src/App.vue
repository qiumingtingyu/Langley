<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import {
  ArrowUp,
  ChevronRight,
  MessageCircleMore,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Trash2,
} from "lucide-vue-next";

import { Button } from "@/components/ui/button";
import MessageContent from "@/components/MessageContent.vue";

type RunStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";

interface Conversation {
  id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
}

interface Message {
  id: number;
  sequence_no: number;
  role: "USER" | "ASSISTANT";
  content: string;
  run_id: number | null;
  regenerated_from_message_id: number | null;
  created_at: string;
}

interface Run {
  id: number;
  input_message_id: number;
  attempt_no: number;
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  error_code: string | null;
}

interface PendingCommand {
  conversationId: number;
  label: string;
  path: string;
  payload: Record<string, string>;
}

interface StreamState {
  conversationId: number;
  runId: number;
  viewRevision: number;
  content: string;
}

const ERROR_MESSAGES: Record<string, string> = {
  ACTIVE_RUN_EXISTS: "当前会话正在回答，请先停止当前回答后再操作。",
  CLIENT_REQUEST_ID_REUSED: "该请求与已有操作不一致，请重新提交。",
  CONVERSATION_NOT_FOUND: "会话不存在或已删除。",
  LLM_PROVIDER_FAILED: "模型服务暂时不可用，请稍后重试。",
  REGENERATE_NOT_ALLOWED: "当前消息暂时无法重新生成。",
  RETRY_NOT_ALLOWED: "当前消息暂时无法重试。",
  RUN_NOT_CANCELLABLE: "该回答已经结束，无法停止。",
  VALIDATION_ERROR: "输入内容不符合要求。",
};

const conversations = ref<Conversation[]>([]);
const selectedConversationId = ref<number | null>(null);
const messages = ref<Message[]>([]);
const latestRun = ref<Run | null>(null);
const composerContent = ref("");
const busyAction = ref<string | null>(null);
const isLoading = ref(true);
const requestError = ref<string | null>(null);
const pendingNetworkCommand = ref<PendingCommand | null>(null);
const streamState = ref<StreamState | null>(null);

let viewRevision = 0;
let readController: AbortController | null = null;
let eventSource: EventSource | null = null;

const selectedConversation = computed(() =>
  conversations.value.find((conversation) => conversation.id === selectedConversationId.value),
);
const hasActiveRun = computed(() => latestRun.value !== null && isActive(latestRun.value));
const runFailureMessage = computed(
  () => ERROR_MESSAGES[latestRun.value?.error_code ?? ""] ?? "回答未完成，请稍后重试。",
);
const hasCurrentStream = computed(
  () =>
    streamState.value?.conversationId === selectedConversationId.value &&
    streamState.value.viewRevision === viewRevision &&
    streamState.value.runId === latestRun.value?.id,
);

function conversationTitle(conversation: Conversation): string {
  return conversation.title ?? "未命名会话";
}

function isActive(run: Run): boolean {
  return run.status === "PENDING" || run.status === "RUNNING";
}

function isCurrentView(conversationId: number, revision: number): boolean {
  return selectedConversationId.value === conversationId && viewRevision === revision;
}

function errorMessage(payload: unknown): string {
  const code =
    typeof payload === "object" && payload !== null && "detail" in payload
      ? (payload.detail as { code?: string }).code
      : undefined;
  return code === undefined
    ? "请求未完成，请稍后重试。"
    : (ERROR_MESSAGES[code] ?? "请求未完成，请稍后重试。");
}

function closeStream(): void {
  eventSource?.close();
  eventSource = null;
  streamState.value = null;
}

function beginView(conversationId: number | null): number {
  viewRevision += 1;
  readController?.abort();
  closeStream();
  selectedConversationId.value = conversationId;
  messages.value = [];
  latestRun.value = null;
  busyAction.value = null;
  requestError.value = null;
  pendingNetworkCommand.value = null;
  return viewRevision;
}

function addMessage(message: Message): void {
  if (!messages.value.some((item) => item.id === message.id)) {
    messages.value = [...messages.value, message];
  }
}

async function loadConversation(conversationId: number, revision = viewRevision): Promise<void> {
  readController?.abort();
  const controller = new AbortController();
  readController = controller;
  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {
      signal: controller.signal,
    });
    const payload = (await response.json()) as { messages: Message[]; latest_run: Run | null };
    if (!isCurrentView(conversationId, revision)) return;
    if (!response.ok) throw new Error(errorMessage(payload));

    messages.value = payload.messages;
    latestRun.value = payload.latest_run;
    if (payload.latest_run !== null && isActive(payload.latest_run)) {
      observeRun(conversationId, payload.latest_run, revision);
    } else {
      closeStream();
    }
  } finally {
    if (readController === controller) readController = null;
  }
}

async function refreshFacts(preferredConversationId?: number, revision = viewRevision): Promise<void> {
  let currentRevision = revision;
  isLoading.value = true;
  try {
    const response = await fetch("/api/conversations");
    const payload = (await response.json()) as Conversation[];
    if (revision !== viewRevision) return;
    if (!response.ok) throw new Error(errorMessage(payload));

    conversations.value = payload;
    const conversationId =
      preferredConversationId ??
      (payload.some((conversation) => conversation.id === selectedConversationId.value)
        ? selectedConversationId.value
        : payload[0]?.id) ??
      null;
    currentRevision =
      conversationId === selectedConversationId.value ? viewRevision : beginView(conversationId);
    if (conversationId !== null) await loadConversation(conversationId, currentRevision);
  } catch (error) {
    if (currentRevision === viewRevision) {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  } finally {
    if (currentRevision === viewRevision) isLoading.value = false;
  }
}

async function selectConversation(conversationId: number): Promise<void> {
  const revision = beginView(conversationId);
  isLoading.value = true;
  try {
    await loadConversation(conversationId, revision);
  } catch (error) {
    if (isCurrentView(conversationId, revision)) {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  } finally {
    if (isCurrentView(conversationId, revision)) isLoading.value = false;
  }
}

function observeRun(
  conversationId: number,
  run: Run,
  revision = viewRevision,
  reconnected = false,
): void {
  if (!isCurrentView(conversationId, revision) || !isActive(run)) return;

  closeStream();
  streamState.value = { conversationId, runId: run.id, viewRevision: revision, content: "" };
  const source = new EventSource(`/api/runs/${run.id}/events`);
  eventSource = source;

  source.addEventListener("run.started", () => {
    if (isCurrentView(conversationId, revision) && latestRun.value?.id === run.id) {
      latestRun.value = { ...latestRun.value, status: "RUNNING" };
    }
  });
  source.addEventListener("message.delta", (event) => {
    if (!isCurrentView(conversationId, revision) || streamState.value?.runId !== run.id) return;
    try {
      const payload = JSON.parse((event as MessageEvent<string>).data) as {
        run_id: number;
        delta: string;
      };
      if (payload.run_id === run.id) {
        streamState.value = { ...streamState.value, content: streamState.value.content + payload.delta };
      }
    } catch {
      void refreshRun(conversationId, run.id, revision, reconnected);
    }
  });
  for (const eventName of ["run.succeeded", "run.failed", "run.cancelled"]) {
    source.addEventListener(eventName, () => {
      if (isCurrentView(conversationId, revision)) {
        source.close();
        void refreshRun(conversationId, run.id, revision, reconnected);
      }
    });
  }
  source.onerror = () => void refreshRun(conversationId, run.id, revision, reconnected);
}

async function refreshRun(
  conversationId: number,
  runId: number,
  revision: number,
  reconnected = false,
): Promise<void> {
  if (!isCurrentView(conversationId, revision) || latestRun.value?.id !== runId) return;
  closeStream();
  try {
    const response = await fetch(`/api/runs/${runId}`);
    const payload = (await response.json()) as { run: Run; assistant_message: Message | null };
    if (!isCurrentView(conversationId, revision) || latestRun.value?.id !== runId) return;
    if (!response.ok) throw new Error(errorMessage(payload));

    latestRun.value = payload.run;
    if (isActive(payload.run)) {
      if (reconnected) {
        requestError.value = "实时连接已断开，请刷新恢复。";
      } else {
        observeRun(conversationId, payload.run, revision, true);
      }
      return;
    }

    if (payload.assistant_message?.run_id === runId) addMessage(payload.assistant_message);
    closeStream();
    await loadConversation(conversationId, revision);
  } catch (error) {
    if (isCurrentView(conversationId, revision)) {
      requestError.value =
        reconnected ? "实时连接已断开，请刷新恢复。" : error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  }
}

async function submitCommand(command: PendingCommand): Promise<void> {
  if (selectedConversationId.value !== command.conversationId) return;
  const revision = viewRevision;
  busyAction.value = command.label;
  try {
    const response = await fetch(command.path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(command.payload),
    });
    const payload = (await response.json()) as { user_message?: Message; run?: Run };
    if (!isCurrentView(command.conversationId, revision)) return;
    if (!response.ok || payload.user_message === undefined || payload.run === undefined) {
      throw new Error(errorMessage(payload));
    }

    pendingNetworkCommand.value = null;
    composerContent.value = "";
    addMessage(payload.user_message);
    latestRun.value = payload.run;
    if (isActive(payload.run)) observeRun(command.conversationId, payload.run, revision);
    else await loadConversation(command.conversationId, revision);
  } catch (error) {
    if (!isCurrentView(command.conversationId, revision)) return;
    if (error instanceof TypeError) {
      pendingNetworkCommand.value = command;
      requestError.value = "网络请求失败，可重试相同操作。";
    } else {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  } finally {
    if (isCurrentView(command.conversationId, revision)) busyAction.value = null;
  }
}

function sendQuestion(): void {
  if (selectedConversationId.value === null || !composerContent.value.trim()) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在生成…",
    path: `/api/conversations/${selectedConversationId.value}/messages`,
    payload: { content: composerContent.value, client_request_id: crypto.randomUUID() },
  });
}

function retryAnswer(): void {
  if (selectedConversationId.value === null) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在重试回答…",
    path: `/api/conversations/${selectedConversationId.value}/retry`,
    payload: { client_request_id: crypto.randomUUID() },
  });
}

function regenerateAnswer(): void {
  if (selectedConversationId.value === null) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在重新生成…",
    path: `/api/conversations/${selectedConversationId.value}/regenerate`,
    payload: { client_request_id: crypto.randomUUID() },
  });
}

function retryNetworkRequest(): void {
  if (pendingNetworkCommand.value !== null) void submitCommand(pendingNetworkCommand.value);
}

async function stopAnswer(): Promise<void> {
  if (selectedConversationId.value === null || latestRun.value === null || !hasActiveRun.value) return;
  const conversationId = selectedConversationId.value;
  const runId = latestRun.value.id;
  const revision = viewRevision;
  busyAction.value = "正在停止…";
  try {
    const response = await fetch(`/api/runs/${runId}/cancel`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorMessage(payload));
    if (isCurrentView(conversationId, revision) && latestRun.value?.id === runId) {
      await refreshRun(conversationId, runId, revision);
    }
  } catch (error) {
    if (isCurrentView(conversationId, revision)) {
      requestError.value = error instanceof Error ? error.message : "请求未完成，请稍后重试。";
    }
  } finally {
    if (isCurrentView(conversationId, revision)) busyAction.value = null;
  }
}

async function createConversation(): Promise<void> {
  const revision = viewRevision;
  busyAction.value = "正在新建会话…";
  try {
    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const conversation = (await response.json()) as Conversation;
    if (!response.ok) throw new Error(errorMessage(conversation));
    if (revision === viewRevision) await refreshFacts(conversation.id, revision);
  } catch (error) {
    if (revision === viewRevision) {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  } finally {
    if (revision === viewRevision) busyAction.value = null;
  }
}

async function renameConversation(): Promise<void> {
  const conversation = selectedConversation.value;
  if (conversation === undefined) return;
  const title = window.prompt("请输入新的会话名称", conversationTitle(conversation));
  if (title === null) return;

  const revision = viewRevision;
  try {
    const response = await fetch(`/api/conversations/${conversation.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!response.ok) throw new Error(errorMessage(await response.json()));
    if (isCurrentView(conversation.id, revision)) {
      conversations.value = conversations.value.map((item) =>
        item.id === conversation.id ? { ...item, title } : item,
      );
    }
  } catch (error) {
    if (isCurrentView(conversation.id, revision)) {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  }
}

async function deleteSelectedConversation(): Promise<void> {
  const conversation = selectedConversation.value;
  if (
    conversation === undefined ||
    !window.confirm(`确定删除“${conversationTitle(conversation)}”吗？历史消息将保留，但不会再显示此会话。`)
  ) return;

  const revision = viewRevision;
  try {
    const response = await fetch(`/api/conversations/${conversation.id}`, { method: "DELETE" });
    if (!response.ok) throw new Error(errorMessage(await response.json()));
    if (!isCurrentView(conversation.id, revision)) return;

    conversations.value = conversations.value.filter((item) => item.id !== conversation.id);
    const nextConversationId = conversations.value[0]?.id ?? null;
    const nextRevision = beginView(nextConversationId);
    if (nextConversationId !== null) await loadConversation(nextConversationId, nextRevision);
  } catch (error) {
    if (isCurrentView(conversation.id, revision)) {
      requestError.value = error instanceof Error ? error.message : "网络或服务暂时不可用，请稍后重试。";
    }
  }
}

onMounted(() => void refreshFacts());
onBeforeUnmount(() => {
  readController?.abort();
  closeStream();
});
</script>

<template>
  <main class="flex min-h-screen bg-stone-50 text-slate-800">
    <aside class="flex w-72 shrink-0 flex-col border-r border-stone-200 bg-stone-100/70 p-4">
      <div class="mb-7 flex items-center gap-2 px-1 text-sm font-semibold tracking-tight text-slate-900">
        <span class="flex size-7 items-center justify-center rounded-md bg-slate-900 text-white">
          <Sparkles
            :size="15"
            aria-hidden="true"
          />
        </span>
        Langley
      </div>

      <Button
        class="mb-5 w-full justify-start"
        :disabled="busyAction !== null"
        @click="createConversation"
      >
        <Plus
          :size="16"
          aria-hidden="true"
        />
        新建会话
      </Button>

      <div class="mb-2 px-2 text-[11px] font-semibold uppercase tracking-[0.13em] text-slate-500">
        会话列表
      </div>
      <nav
        class="min-h-0 flex-1 space-y-1 overflow-y-auto"
        aria-label="会话列表"
      >
        <button
          v-for="conversation in conversations"
          :key="conversation.id"
          class="group flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors"
          :class="conversation.id === selectedConversationId ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-600 hover:bg-stone-200/70 hover:text-slate-950'"
          type="button"
          @click="selectConversation(conversation.id)"
        >
          <MessageCircleMore
            :size="15"
            aria-hidden="true"
            class="shrink-0 opacity-70"
          />
          <span class="truncate">{{ conversationTitle(conversation) }}</span>
          <ChevronRight
            v-if="conversation.id === selectedConversationId"
            :size="14"
            aria-hidden="true"
            class="ml-auto text-slate-400"
          />
        </button>
        <p
          v-if="conversations.length === 0 && !isLoading"
          class="px-2 py-3 text-sm leading-6 text-slate-500"
        >
          新建一个会话后即可开始。
        </p>
      </nav>
    </aside>

    <section class="flex min-w-0 flex-1 flex-col">
      <header class="flex h-16 shrink-0 items-center justify-between border-b border-stone-200 bg-stone-50/90 px-8">
        <div>
          <p class="text-sm font-semibold text-slate-900">
            {{ selectedConversation ? conversationTitle(selectedConversation) : "Langley" }}
          </p>
          <p class="mt-0.5 text-xs text-slate-500">
            {{ selectedConversation ? "学习对话" : "请选择或新建会话" }}
          </p>
        </div>
        <div class="flex items-center gap-1">
          <Button
            v-if="selectedConversation"
            variant="ghost"
            size="icon"
            aria-label="重命名会话"
            :disabled="busyAction !== null"
            @click="renameConversation"
          >
            <Pencil
              :size="16"
              aria-hidden="true"
            />
          </Button>
          <Button
            v-if="selectedConversation"
            variant="ghost"
            size="icon"
            aria-label="删除会话"
            :disabled="busyAction !== null"
            @click="deleteSelectedConversation"
          >
            <Trash2
              :size="16"
              aria-hidden="true"
            />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="刷新会话"
            :disabled="isLoading"
            @click="refreshFacts()"
          >
            <RefreshCw
              :size="17"
              :class="{ 'animate-spin': isLoading }"
              aria-hidden="true"
            />
          </Button>
        </div>
      </header>

      <div class="flex min-h-0 flex-1 flex-col">
        <div class="mx-auto flex w-full max-w-3xl flex-1 flex-col px-8 py-10">
          <p
            v-if="requestError"
            role="alert"
            class="mb-5 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
          >
            {{ requestError }}
            <Button
              v-if="pendingNetworkCommand"
              variant="outline"
              size="small"
              class="ml-3"
              @click="retryNetworkRequest"
            >
              重试请求
            </Button>
          </p>

          <div
            v-if="!selectedConversation && !isLoading"
            class="flex flex-1 items-center justify-center"
          >
            <div class="max-w-sm text-center">
              <h1 class="text-xl font-semibold tracking-tight text-slate-900">
                开始学习对话
              </h1>
              <p class="mt-2 text-sm leading-6 text-slate-500">
                新建一个会话，开始提问。
              </p>
            </div>
          </div>

          <div
            v-else
            class="space-y-7"
          >
            <template
              v-for="message in messages"
              :key="message.id"
            >
              <article
                v-if="message.role === 'USER'"
                class="flex justify-end"
              >
                <div class="max-w-[82%] break-words whitespace-pre-wrap rounded-2xl rounded-br-md bg-slate-900 px-4 py-3 text-sm leading-6 text-white shadow-sm">
                  {{ message.content }}
                </div>
              </article>
              <article
                v-else
                class="max-w-[88%] text-[15px] leading-7 text-slate-700"
              >
                <p class="mb-1 text-[11px] font-semibold tracking-[0.13em] text-slate-400">
                  Langley
                </p>
                <MessageContent :content="message.content" />
              </article>
            </template>

            <article
              v-if="hasCurrentStream"
              class="max-w-[88%] text-[15px] leading-7 text-slate-700"
              aria-live="polite"
            >
              <p class="mb-1 text-[11px] font-semibold tracking-[0.13em] text-slate-400">
                Langley
              </p>
              <MessageContent :content="streamState?.content || '正在生成…'" />
            </article>

            <div
              v-if="messages.length === 0 && selectedConversation && !isLoading"
              class="py-16 text-center text-sm text-slate-500"
            >
              这个会话已准备好，随时可以开始提问。
            </div>

            <div
              v-if="busyAction"
              class="flex items-center gap-2 text-sm text-slate-500"
              aria-live="polite"
            >
              <span
                class="size-2 animate-pulse rounded-full bg-slate-400"
                aria-hidden="true"
              />
              {{ busyAction }}
            </div>

            <div
              v-else-if="hasActiveRun"
              class="flex items-center gap-3 text-sm text-slate-500"
              aria-live="polite"
            >
              <span
                class="size-2 animate-pulse rounded-full bg-slate-400"
                aria-hidden="true"
              />
              {{ latestRun?.status === "PENDING" ? "正在生成…" : "正在处理…" }}
              <Button
                variant="outline"
                size="small"
                :disabled="busyAction !== null"
                @click="stopAnswer"
              >
                停止
              </Button>
            </div>

            <div
              v-else-if="latestRun?.status === 'FAILED'"
              class="flex items-center justify-between rounded-lg border border-rose-200 bg-rose-50 px-4 py-3"
            >
              <div>
                <p class="text-sm font-medium text-rose-950">
                  回答失败
                </p>
                <p class="mt-0.5 text-xs text-rose-700">
                  {{ runFailureMessage }}
                </p>
              </div>
              <Button
                variant="outline"
                size="small"
                @click="retryAnswer"
              >
                <RotateCcw
                  :size="14"
                  aria-hidden="true"
                />
                重试
              </Button>
            </div>

            <div
              v-else-if="latestRun?.status === 'CANCELLED'"
              class="flex items-center justify-between rounded-lg border border-stone-200 bg-stone-100 px-4 py-3"
            >
              <div>
                <p class="text-sm font-medium text-slate-900">
                  已停止回答
                </p>
                <p class="mt-0.5 text-xs text-slate-600">
                  问题已保存，可以重新尝试。
                </p>
              </div>
              <Button
                variant="outline"
                size="small"
                @click="retryAnswer"
              >
                <RotateCcw
                  :size="14"
                  aria-hidden="true"
                />
                重试
              </Button>
            </div>

            <div
              v-else-if="latestRun?.status === 'SUCCEEDED'"
              class="flex items-center justify-between border-t border-stone-200 pt-4"
            >
              <p class="text-xs text-slate-500">
                回答已保存
              </p>
              <Button
                variant="ghost"
                size="small"
                @click="regenerateAnswer"
              >
                <RotateCcw
                  :size="14"
                  aria-hidden="true"
                />
                重新生成
              </Button>
            </div>
          </div>

          <form
            v-if="selectedConversation"
            class="mt-8 border-t border-stone-200 pt-6"
            @submit.prevent="sendQuestion"
          >
            <label
              class="sr-only"
              for="question"
            >输入问题</label>
            <div class="rounded-xl border border-stone-300 bg-white p-2 shadow-sm focus-within:border-slate-400 focus-within:ring-2 focus-within:ring-slate-200">
              <textarea
                id="question"
                v-model="composerContent"
                class="block min-h-24 w-full resize-none border-0 bg-transparent px-2 py-1.5 text-sm leading-6 text-slate-800 outline-none placeholder:text-slate-400"
                :disabled="busyAction !== null || hasActiveRun"
                placeholder="输入你的学习问题…"
                @keydown.meta.enter.prevent="sendQuestion"
                @keydown.ctrl.enter.prevent="sendQuestion"
              />
              <div class="flex items-center justify-between px-1 pt-1">
                <span class="text-xs text-slate-400">Ctrl/⌘ + Enter 发送</span>
                <Button
                  type="submit"
                  size="icon"
                  aria-label="发送问题"
                  :disabled="busyAction !== null || hasActiveRun || !composerContent.trim()"
                >
                  <ArrowUp
                    :size="16"
                    aria-hidden="true"
                  />
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </section>
  </main>
</template>
