<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import AppSidebar from "@/components/AppSidebar.vue";
import ChatWorkspace from "@/components/ChatWorkspace.vue";
import KnowledgePage from "@/KnowledgePage.vue";
import MemoryPage from "@/MemoryPage.vue";
import type { ActiveView, Conversation, GroundingPolicy, KnowledgeBase, Message, Run, StreamState } from "@/types";

interface CommandScope {
  knowledgeBaseId: number | null;
  groundingPolicy: GroundingPolicy;
}

interface PendingCommand {
  conversationId: number;
  label: string;
  path: string;
  payload: Record<string, unknown>;
  scope: CommandScope;
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
const knowledgeBases = ref<KnowledgeBase[]>([]);
const selectedKnowledgeBaseId = ref<number | null>(null);
const groundingPolicy = ref<GroundingPolicy>("AUTO");
const isLoadingKnowledgeBases = ref(false);
const knowledgeBaseLoadError = ref<string | null>(null);
const busyAction = ref<string | null>(null);
const isLoading = ref(true);
const requestError = ref<string | null>(null);
const pendingNetworkCommand = ref<PendingCommand | null>(null);
const activeCommandScope = ref<CommandScope | null>(null);
const streamState = ref<StreamState | null>(null);
const activeView = ref<ActiveView>("chat");
const memoryNotice = ref<string | null>(null);
const memoryPage = ref<{ load(): Promise<void> } | null>(null);
const memoryUpdated = ref(false);

let viewRevision = 0;
let readController: AbortController | null = null;
let eventSource: EventSource | null = null;
let memoryEventSource: EventSource | null = null;
let memoryNoticeTimer: ReturnType<typeof setTimeout> | null = null;

const selectedConversation = computed(() =>
  conversations.value.find((conversation) => conversation.id === selectedConversationId.value),
);
const hasActiveRun = computed(() => latestRun.value !== null && isActive(latestRun.value));
const displayedKnowledgeScope = computed<CommandScope>(() => {
  if (activeCommandScope.value !== null) return activeCommandScope.value;
  if (pendingNetworkCommand.value !== null) return pendingNetworkCommand.value.scope;
  if (hasActiveRun.value && latestRun.value !== null) return scopeFromRun(latestRun.value);
  return { knowledgeBaseId: selectedKnowledgeBaseId.value, groundingPolicy: groundingPolicy.value };
});
const isKnowledgeScopeLocked = computed(
  () => busyAction.value !== null || hasActiveRun.value || pendingNetworkCommand.value !== null,
);
const runFailureMessage = computed(
  () => ERROR_MESSAGES[latestRun.value?.error_code ?? ""] ?? "回答未完成，请稍后重试。",
);
const hasCurrentStream = computed(
  () =>
    streamState.value?.conversationId === selectedConversationId.value &&
    streamState.value.viewRevision === viewRevision &&
    streamState.value.runId === latestRun.value?.id,
);

function isActive(run: Run): boolean {
  return run.status === "PENDING" || run.status === "RUNNING";
}

function scopeFromRun(run: Run): CommandScope {
  return { knowledgeBaseId: run.knowledge_base_id, groundingPolicy: run.grounding_policy };
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

function showMemoryNotice(message: string): void {
  memoryNotice.value = message;
  if (memoryNoticeTimer !== null) clearTimeout(memoryNoticeTimer);
  memoryNoticeTimer = setTimeout(() => { memoryNotice.value = null; }, 4000);
}

function openMemoryView(): void {
  activeView.value = "memory";
  memoryUpdated.value = false;
}

function openKnowledgeView(): void {
  activeView.value = "knowledge";
}

function enterChatView(): void {
  const shouldRefreshKnowledgeBases = activeView.value !== "chat";
  activeView.value = "chat";
  if (shouldRefreshKnowledgeBases) void loadKnowledgeBases();
}

function openChatView(): void {
  enterChatView();
}

function updatedNotice(payload: {
  created_count?: number;
  changed_count?: number;
  forgotten_count?: number;
}): string {
  const created = payload.created_count ?? 0;
  const changed = payload.changed_count ?? 0;
  const forgotten = payload.forgotten_count ?? 0;
  if (created > 0 && changed === 0 && forgotten === 0) return "长期记忆已保存。";
  if (changed > 0 && created === 0 && forgotten === 0) return "长期记忆已更新。";
  if (forgotten > 0 && created === 0 && changed === 0) return "长期记忆已移除。";
  return "长期记忆已更新。";
}

function observeMemoryEvents(): void {
  memoryEventSource?.close();
  const source = new EventSource("/api/memory-events");
  memoryEventSource = source;
  source.addEventListener("memory.updated", (event) => {
    const payload = JSON.parse((event as MessageEvent<string>).data) as {
      user_requested_memory_action?: boolean;
      created_count?: number;
      changed_count?: number;
      forgotten_count?: number;
    };
    if (payload.user_requested_memory_action) showMemoryNotice(updatedNotice(payload));
    else memoryUpdated.value = true;
    if (activeView.value === "memory") void memoryPage.value?.load();
  });
  source.addEventListener("memory.no_change", () => showMemoryNotice("本次未对长期记忆做出修改。"));
  source.addEventListener("memory.retry_pending", () => showMemoryNotice("长期记忆同步暂时未完成，后续交互时会再次尝试。"));
  source.addEventListener("memory.not_saved", () => showMemoryNotice("本次内容未保存为长期记忆。"));
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
  activeCommandScope.value = null;
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

async function loadKnowledgeBases(): Promise<void> {
  isLoadingKnowledgeBases.value = true;
  try {
    const response = await fetch("/api/knowledge-bases");
    const payload = (await response.json()) as KnowledgeBase[];
    if (!response.ok) throw new Error("资料暂时不可用，请稍后重试。");
    knowledgeBases.value = payload;
    if (!payload.some((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId.value)) {
      updateKnowledgeBaseId(null);
    }
    knowledgeBaseLoadError.value = null;
  } catch {
    knowledgeBases.value = [];
    updateKnowledgeBaseId(null);
    knowledgeBaseLoadError.value = "资料暂时不可用，不影响普通对话。";
  } finally {
    isLoadingKnowledgeBases.value = false;
  }
}

function updateKnowledgeBaseId(knowledgeBaseId: number | null): void {
  selectedKnowledgeBaseId.value = knowledgeBaseId;
  if (knowledgeBaseId === null) groundingPolicy.value = "AUTO";
}

function updateGroundingPolicy(policy: GroundingPolicy): void {
  groundingPolicy.value = selectedKnowledgeBaseId.value === null ? "AUTO" : policy;
}

async function selectConversation(conversationId: number): Promise<void> {
  enterChatView();
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
  activeCommandScope.value = command.scope;
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
    if (isCurrentView(command.conversationId, revision)) {
      activeCommandScope.value = null;
      busyAction.value = null;
    }
  }
}

function sendQuestion(): void {
  if (selectedConversationId.value === null || !composerContent.value.trim()) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在生成…",
    path: `/api/conversations/${selectedConversationId.value}/messages`,
    payload: {
      content: composerContent.value,
      client_request_id: crypto.randomUUID(),
      knowledge_base_id: selectedKnowledgeBaseId.value,
      grounding_policy: groundingPolicy.value,
    },
    scope: { knowledgeBaseId: selectedKnowledgeBaseId.value, groundingPolicy: groundingPolicy.value },
  });
}

function retryAnswer(): void {
  if (selectedConversationId.value === null || latestRun.value === null) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在重试回答…",
    path: `/api/conversations/${selectedConversationId.value}/retry`,
    payload: { client_request_id: crypto.randomUUID() },
    scope: scopeFromRun(latestRun.value),
  });
}

function regenerateAnswer(): void {
  if (selectedConversationId.value === null || latestRun.value === null) return;
  void submitCommand({
    conversationId: selectedConversationId.value,
    label: "正在重新生成…",
    path: `/api/conversations/${selectedConversationId.value}/regenerate`,
    payload: { client_request_id: crypto.randomUUID() },
    scope: scopeFromRun(latestRun.value),
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

async function renameConversation(title: string): Promise<void> {
  const conversation = selectedConversation.value;
  if (conversation === undefined) return;

  const revision = viewRevision;
  busyAction.value = "正在重命名…";
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
  } finally {
    if (busyAction.value === "正在重命名…") busyAction.value = null;
  }
}

async function deleteSelectedConversation(): Promise<void> {
  const conversation = selectedConversation.value;
  if (conversation === undefined) return;

  const revision = viewRevision;
  busyAction.value = "正在删除…";
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
  } finally {
    if (busyAction.value === "正在删除…") busyAction.value = null;
  }
}

onMounted(() => {
  void refreshFacts().finally(() => void loadKnowledgeBases());
  observeMemoryEvents();
});
onBeforeUnmount(() => {
  readController?.abort();
  closeStream();
  memoryEventSource?.close();
  if (memoryNoticeTimer !== null) clearTimeout(memoryNoticeTimer);
});
</script>

<template>
  <main class="relative flex h-screen min-h-0 min-w-0 flex-col overflow-hidden bg-canvas text-foreground md:flex-row">
    <AppSidebar
      :conversations="conversations"
      :selected-conversation-id="selectedConversationId"
      :active-view="activeView"
      :memory-updated="memoryUpdated"
      :busy="busyAction !== null"
      :loading="isLoading"
      @create="createConversation"
      @select="selectConversation"
      @open-chat="openChatView"
      @open-knowledge="openKnowledgeView"
      @open-memory="openMemoryView"
    />

    <p
      v-if="memoryNotice"
      role="status"
      class="absolute right-5 top-16 z-10 max-w-sm rounded-md border border-strong-border bg-surface px-3 py-2 text-sm text-body shadow-lg md:top-5"
    >
      {{ memoryNotice }}
    </p>
    <ChatWorkspace
      v-if="activeView === 'chat'"
      v-model:composer-content="composerContent"
      :knowledge-bases="knowledgeBases"
      :knowledge-base-id="displayedKnowledgeScope.knowledgeBaseId"
      :grounding-policy="displayedKnowledgeScope.groundingPolicy"
      :is-knowledge-scope-locked="isKnowledgeScopeLocked"
      :is-loading-knowledge-bases="isLoadingKnowledgeBases"
      :knowledge-base-load-error="knowledgeBaseLoadError"
      :selected-conversation="selectedConversation ?? null"
      :messages="messages"
      :latest-run="latestRun"
      :stream-content="hasCurrentStream ? streamState?.content ?? '' : null"
      :busy-action="busyAction"
      :is-loading="isLoading"
      :request-error="requestError"
      :has-pending-network-command="pendingNetworkCommand !== null"
      :has-active-run="hasActiveRun"
      :run-failure-message="runFailureMessage"
      @refresh="refreshFacts()"
      @rename="renameConversation"
      @delete="deleteSelectedConversation"
      @retry-network="retryNetworkRequest"
      @stop="stopAnswer"
      @retry="retryAnswer"
      @regenerate="regenerateAnswer"
      @send="sendQuestion"
      @update:knowledge-base-id="updateKnowledgeBaseId"
      @update:grounding-policy="updateGroundingPolicy"
      @retry-knowledge-bases="loadKnowledgeBases"
    />
    <section
      v-else-if="activeView === 'memory'"
      class="min-w-0 flex-1 overflow-y-auto bg-workspace"
    >
      <MemoryPage
        ref="memoryPage"
        @notice="showMemoryNotice"
      />
    </section>
    <section
      v-else
      class="min-h-0 min-w-0 flex-1 overflow-y-auto bg-workspace lg:overflow-hidden"
    >
      <KnowledgePage @notice="showMemoryNotice" />
    </section>
  </main>
</template>
