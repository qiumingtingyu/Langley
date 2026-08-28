<script setup lang="ts">
import { ArrowUp, Pencil, RefreshCw, RotateCcw, Trash2 } from "lucide-vue-next";
import { ref, watch } from "vue";

import EvidenceSheet from "@/components/EvidenceSheet.vue";
import MessageContent from "@/components/MessageContent.vue";
import { Button } from "@/components/ui/button";
import type { Conversation, Message, MessageCitation, Run } from "@/types";

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
}>();

const composerContent = defineModel<string>("composerContent", { required: true });
const selectedCitation = ref<MessageCitation | null>(null);

const emit = defineEmits<{
  refresh: [];
  rename: [];
  delete: [];
  retryNetwork: [];
  stop: [];
  retry: [];
  regenerate: [];
  send: [];
}>();

function conversationTitle(conversation: Conversation): string {
  return conversation.title ?? "未命名会话";
}

watch(
  () => props.selectedConversation?.id,
  () => {
    selectedCitation.value = null;
  },
);
</script>

<template>
  <section class="flex min-w-0 flex-1 flex-col bg-workspace">
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
          @click="emit('rename')"
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
          @click="emit('delete')"
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

    <div class="flex min-h-0 flex-1 overflow-y-auto">
      <div class="mx-auto flex w-full max-w-[52rem] flex-1 flex-col px-5 py-8 sm:px-8 sm:py-10 lg:px-12">
        <p
          v-if="requestError"
          role="alert"
          class="mb-6 rounded-md border border-amber-300/70 bg-amber-50 px-3 py-2 text-sm text-amber-950"
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
            class="border-y border-border py-14 text-center text-sm text-muted-foreground"
          >
            这个会话已准备好，随时可以开始提问。
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
            class="flex items-center justify-between gap-4 rounded-md border border-rose-300/70 bg-rose-50 px-4 py-3"
          >
            <div>
              <p class="text-sm font-medium text-rose-950">
                回答失败
              </p>
              <p class="mt-0.5 text-xs text-rose-800">
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

        <form
          v-if="selectedConversation"
          class="mt-10 border-t border-border pt-6"
          @submit.prevent="emit('send')"
        >
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
            <div class="flex items-center justify-between gap-3 px-1 pt-1">
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
        </form>
      </div>
    </div>

    <EvidenceSheet
      :citation="selectedCitation"
      @close="selectedCitation = null"
    />
  </section>
</template>
