<script setup lang="ts">
import { ArrowUp, Pencil, RefreshCw, RotateCcw, Trash2 } from "lucide-vue-next";
import {
  DialogContent,
  DialogDescription,
  DialogOverlay,
  DialogPortal,
  DialogRoot,
  DialogTitle,
} from "reka-ui";
import { computed, ref, watch } from "vue";

import EvidenceSheet from "@/components/EvidenceSheet.vue";
import MessageContent from "@/components/MessageContent.vue";
import { Button } from "@/components/ui/button";
import type { Conversation, GroundingPolicy, KnowledgeBase, Message, MessageCitation, Run } from "@/types";

const props = defineProps<{
  selectedConversation: Conversation | null;
  messages: Message[];
  latestRun: Run | null;
  streamContent: string | null;
  busyAction: string | null;
  isLoading: boolean;
  requestError: string | null;
  hasPendingNetworkCommand: boolean;
  hasActiveRun: boolean;
  runFailureMessage: string;
  knowledgeBases: KnowledgeBase[];
  knowledgeBaseId: number | null;
  groundingPolicy: GroundingPolicy;
  isKnowledgeScopeLocked: boolean;
  isLoadingKnowledgeBases: boolean;
  knowledgeBaseLoadError: string | null;
}>();

const composerContent = defineModel<string>("composerContent", { required: true });
const selectedCitation = ref<MessageCitation | null>(null);
const renameDialogOpen = ref(false);
const deleteDialogOpen = ref(false);
const renameTitle = ref("");
const hasUnavailableKnowledgeBaseName = computed(() =>
  props.isKnowledgeScopeLocked &&
  props.knowledgeBaseId !== null &&
  !props.knowledgeBases.some((knowledgeBase) => knowledgeBase.id === props.knowledgeBaseId),
);

const emit = defineEmits<{
  refresh: [];
  rename: [title: string];
  delete: [];
  retryNetwork: [];
  stop: [];
  retry: [];
  regenerate: [];
  send: [];
  "update:knowledgeBaseId": [value: number | null];
  "update:groundingPolicy": [value: GroundingPolicy];
  retryKnowledgeBases: [];
}>();

function conversationTitle(conversation: Conversation): string {
  return conversation.title ?? "未命名会话";
}

function selectKnowledgeBase(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  emit("update:knowledgeBaseId", value === "" ? null : Number(value));
}

function selectGroundingPolicy(event: Event): void {
  emit("update:groundingPolicy", (event.target as HTMLSelectElement).value as GroundingPolicy);
}

function openRenameDialog(): void {
  if (props.selectedConversation === null) return;
  renameTitle.value = conversationTitle(props.selectedConversation);
  renameDialogOpen.value = true;
}

function submitRename(): void {
  emit("rename", renameTitle.value);
  renameDialogOpen.value = false;
}

function confirmDelete(): void {
  emit("delete");
  deleteDialogOpen.value = false;
}

watch(
  () => props.selectedConversation?.id,
  () => {
    selectedCitation.value = null;
  },
);
</script>

<template>
  <section class="flex min-h-0 min-w-0 flex-1 flex-col bg-workspace">
    <header class="flex min-h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-surface/80 px-5 py-2 sm:px-8">
      <div class="min-w-0">
        <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
          CONVERSATION
        </p>
        <h1 class="mt-0.5 truncate text-sm font-semibold tracking-[-0.01em] text-foreground">
          {{ selectedConversation ? conversationTitle(selectedConversation) : "Langley" }}
        </h1>
      </div>
      <div class="flex shrink-0 items-center gap-0.5">
        <Button
          v-if="selectedConversation"
          variant="ghost"
          size="icon"
          aria-label="重命名会话"
          :disabled="busyAction !== null"
          @click="openRenameDialog"
        >
          <Pencil
            :size="15"
            :stroke-width="1.7"
            aria-hidden="true"
          />
        </Button>
        <Button
          v-if="selectedConversation"
          variant="ghost"
          size="icon"
          aria-label="删除会话"
          :disabled="busyAction !== null"
          @click="deleteDialogOpen = true"
        >
          <Trash2
            :size="15"
            :stroke-width="1.7"
            aria-hidden="true"
          />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="刷新会话"
          :disabled="isLoading"
          @click="emit('refresh')"
        >
          <RefreshCw
            :size="16"
            :stroke-width="1.7"
            :class="{ 'animate-spin': isLoading }"
            aria-hidden="true"
          />
        </Button>
      </div>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <div class="mx-auto flex min-h-full w-full max-w-[52rem] flex-col px-5 py-8 sm:px-8 sm:py-10 lg:px-12">
        <p
          v-if="requestError"
          role="alert"
          class="mb-6 rounded-md border border-warning-border bg-warning-surface px-3 py-2 text-sm text-warning-foreground"
        >
          {{ requestError }}
          <Button
            v-if="hasPendingNetworkCommand"
            variant="outline"
            size="small"
            class="ml-3"
            @click="emit('retryNetwork')"
          >
            重试请求
          </Button>
        </p>

        <div
          v-if="!selectedConversation && !isLoading"
          class="flex flex-1 items-center justify-center py-16"
        >
          <div class="max-w-sm text-center">
            <p class="font-mono text-[10px] tracking-[0.15em] text-muted-light">
              READY
            </p>
            <h2 class="mt-3 text-xl font-semibold tracking-[-0.02em] text-foreground">
              开始学习对话
            </h2>
            <p class="mt-2 text-sm leading-6 text-muted-foreground">
              新建一个会话，开始提问。
            </p>
          </div>
        </div>

        <div
          v-else
          class="space-y-8"
        >
          <template
            v-for="message in messages"
            :key="message.id"
          >
            <article
              v-if="message.role === 'USER'"
              class="flex justify-end"
            >
              <div class="max-w-[82%] break-words whitespace-pre-wrap rounded-md bg-foreground px-4 py-3 text-sm leading-6 text-surface">
                {{ message.content }}
              </div>
            </article>
            <article
              v-else
              class="min-w-0 text-[15px] leading-7 text-body"
            >
              <p class="mb-2 font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
                ASSISTANT
              </p>
              <MessageContent
                :content="message.content"
                :citations="message.citations"
                @select-citation="selectedCitation = $event"
              />
            </article>
          </template>

          <article
            v-if="streamContent !== null"
            class="min-w-0 text-[15px] leading-7 text-body"
            aria-live="polite"
          >
            <p class="mb-2 font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
              ASSISTANT · LIVE
            </p>
            <MessageContent :content="streamContent || '正在生成…'" />
          </article>

          <div
            v-if="messages.length === 0 && selectedConversation && !isLoading"
            class="flex min-h-56 flex-1 items-center justify-center py-14 text-center"
          >
            <div>
              <p class="font-mono text-[10px] font-medium tracking-[0.15em] text-muted-light">
                READY
              </p>
              <h2 class="mt-3 text-xl font-semibold tracking-[-0.02em] text-foreground">
                这个会话已准备好
              </h2>
              <p class="mt-2 text-sm leading-6 text-muted-foreground">
                随时可以开始提问。
              </p>
            </div>
          </div>

          <div
            v-if="busyAction"
            class="flex items-center gap-2 text-sm text-muted-foreground"
            aria-live="polite"
          >
            <span
              class="size-1.5 animate-pulse rounded-full bg-primary"
              aria-hidden="true"
            />
            {{ busyAction }}
          </div>

          <div
            v-else-if="hasActiveRun"
            class="flex items-center gap-3 border-t border-border pt-4 text-sm text-muted-foreground"
            aria-live="polite"
          >
            <span
              class="size-1.5 animate-pulse rounded-full bg-primary"
              aria-hidden="true"
            />
            {{ latestRun?.status === "PENDING" ? "正在生成…" : "正在处理…" }}
            <Button
              variant="outline"
              size="small"
              :disabled="busyAction !== null"
              @click="emit('stop')"
            >
              停止
            </Button>
          </div>

          <div
            v-else-if="latestRun?.status === 'FAILED'"
            class="flex items-center justify-between gap-4 rounded-md border border-danger-border bg-danger-surface px-4 py-3"
          >
            <div>
              <p class="text-sm font-medium text-danger-foreground">
                回答失败
              </p>
              <p class="mt-0.5 text-xs text-danger-foreground">
                {{ runFailureMessage }}
              </p>
            </div>
            <Button
              variant="outline"
              size="small"
              @click="emit('retry')"
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
            class="flex items-center justify-between gap-4 rounded-md border border-border bg-subtle px-4 py-3"
          >
            <div>
              <p class="text-sm font-medium text-foreground">
                已停止回答
              </p>
              <p class="mt-0.5 text-xs text-muted-foreground">
                问题已保存，可以重新尝试。
              </p>
            </div>
            <Button
              variant="outline"
              size="small"
              @click="emit('retry')"
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
            class="flex items-center justify-between border-t border-border pt-4"
          >
            <p class="text-xs text-muted-foreground">
              回答已保存
            </p>
            <Button
              variant="ghost"
              size="small"
              @click="emit('regenerate')"
            >
              <RotateCcw
                :size="14"
                aria-hidden="true"
              />
              重新生成
            </Button>
          </div>
        </div>
      </div>
    </div>

    <form
      v-if="selectedConversation"
      class="shrink-0 border-t border-border bg-workspace px-5 py-4 sm:px-8 sm:py-5 lg:px-12"
      @submit.prevent="emit('send')"
    >
      <div class="mx-auto w-full max-w-[52rem]">
        <label
          class="sr-only"
          for="question"
        >输入问题</label>
        <div class="rounded-lg border border-strong-border bg-surface p-2 transition-colors focus-within:border-primary focus-within:ring-2 focus-within:ring-ring/20">
          <textarea
            id="question"
            v-model="composerContent"
            class="block min-h-24 w-full resize-none border-0 bg-transparent px-2 py-1.5 text-sm leading-6 text-foreground outline-none placeholder:text-muted-light"
            :disabled="busyAction !== null || hasActiveRun"
            placeholder="输入你的学习问题…"
            @keydown.meta.enter.prevent="emit('send')"
            @keydown.ctrl.enter.prevent="emit('send')"
          />
          <div class="flex flex-col gap-2 px-1 pt-2 sm:flex-row sm:items-center sm:justify-between">
            <div class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1.5">
              <label
                for="knowledge-base"
                class="font-mono text-[9px] tracking-[0.1em] text-muted-light"
              >资料范围</label>
              <select
                id="knowledge-base"
                aria-label="资料范围"
                class="min-w-0 max-w-full rounded-sm border border-border bg-surface px-2 py-1 text-xs text-body outline-none focus:border-primary focus:ring-2 focus:ring-ring/20 disabled:cursor-not-allowed disabled:text-muted-light"
                :value="knowledgeBaseId ?? ''"
                :disabled="isKnowledgeScopeLocked || isLoadingKnowledgeBases || knowledgeBaseLoadError !== null"
                @change="selectKnowledgeBase"
              >
                <option
                  v-if="hasUnavailableKnowledgeBaseName"
                  :value="knowledgeBaseId"
                >
                  当前资料（名称暂不可用）
                </option>
                <option value="">
                  不使用资料
                </option>
                <option
                  v-for="knowledgeBase in knowledgeBases"
                  :key="knowledgeBase.id"
                  :value="knowledgeBase.id"
                >
                  {{ knowledgeBase.name }}
                </option>
              </select>
              <template v-if="knowledgeBaseId !== null">
                <label
                  for="grounding-policy"
                  class="font-mono text-[9px] tracking-[0.1em] text-muted-light"
                >依据方式</label>
                <select
                  id="grounding-policy"
                  aria-label="依据方式"
                  class="rounded-sm border border-border bg-surface px-2 py-1 text-xs text-body outline-none focus:border-primary focus:ring-2 focus:ring-ring/20 disabled:cursor-not-allowed disabled:text-muted-light"
                  :value="groundingPolicy"
                  :disabled="isKnowledgeScopeLocked"
                  @change="selectGroundingPolicy"
                >
                  <option value="AUTO">
                    自动参考
                  </option>
                  <option value="REQUIRED">
                    必须依据资料
                  </option>
                </select>
              </template>
              <span
                v-if="isLoadingKnowledgeBases"
                class="text-xs text-muted-light"
              >正在读取资料…</span>
              <span
                v-else-if="knowledgeBaseLoadError"
                class="flex items-center gap-1.5 text-xs text-muted-foreground"
              >
                {{ knowledgeBaseLoadError }}
                <button
                  type="button"
                  class="text-primary-deep underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                  @click="emit('retryKnowledgeBases')"
                >重新读取</button>
              </span>
            </div>
            <span class="font-mono text-[9px] tracking-[0.1em] text-muted-light">COMPOSER · CTRL/⌘ + ENTER</span>
            <Button
              type="submit"
              size="icon"
              aria-label="发送问题"
              :disabled="busyAction !== null || hasActiveRun || !composerContent.trim()"
            >
              <ArrowUp
                :size="16"
                :stroke-width="1.8"
                aria-hidden="true"
              />
            </Button>
          </div>
        </div>
      </div>
    </form>

    <DialogRoot v-model:open="renameDialogOpen">
      <DialogPortal>
        <DialogOverlay class="fixed inset-0 z-40 bg-foreground/20" />
        <DialogContent class="fixed left-1/2 top-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-md border border-strong-border bg-surface p-5 text-foreground shadow-lg outline-none sm:p-6">
          <DialogTitle class="text-base font-semibold">
            重命名会话
          </DialogTitle>
          <DialogDescription class="mt-1 text-sm text-muted-foreground">
            为当前会话设置一个易于识别的名称。
          </DialogDescription>
          <form
            class="mt-5"
            @submit.prevent="submitRename"
          >
            <label
              class="block text-sm font-medium text-body"
              for="rename-conversation-title"
            >会话名称</label>
            <input
              id="rename-conversation-title"
              v-model="renameTitle"
              class="mt-2 w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
              maxlength="255"
              :disabled="busyAction !== null"
            >
            <div class="mt-5 flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                :disabled="busyAction !== null"
                @click="renameDialogOpen = false"
              >
                取消
              </Button>
              <Button
                type="submit"
                :disabled="busyAction !== null"
              >
                保存
              </Button>
            </div>
          </form>
        </DialogContent>
      </DialogPortal>
    </DialogRoot>

    <DialogRoot v-model:open="deleteDialogOpen">
      <DialogPortal>
        <DialogOverlay class="fixed inset-0 z-40 bg-foreground/20" />
        <DialogContent class="fixed left-1/2 top-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-md border border-strong-border bg-surface p-5 text-foreground shadow-lg outline-none sm:p-6">
          <DialogTitle class="text-base font-semibold">
            删除会话？
          </DialogTitle>
          <DialogDescription class="mt-3 text-sm leading-6 text-muted-foreground">
            确定删除“{{ selectedConversation ? conversationTitle(selectedConversation) : '' }}”吗？历史消息将保留，但该会话不会再显示。
          </DialogDescription>
          <div class="mt-5 flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              :disabled="busyAction !== null"
              @click="deleteDialogOpen = false"
            >
              取消
            </Button>
            <Button
              type="button"
              variant="outline"
              class="border-danger-border text-danger-foreground hover:bg-danger-surface"
              :disabled="busyAction !== null"
              @click="confirmDelete"
            >
              删除
            </Button>
          </div>
        </DialogContent>
      </DialogPortal>
    </DialogRoot>

    <EvidenceSheet
      :citation="selectedCitation"
      @close="selectedCitation = null"
    />
  </section>
</template>
