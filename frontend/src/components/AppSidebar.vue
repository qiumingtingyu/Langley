<script setup lang="ts">
import {
  BookOpenText,
  Brain,
  LibraryBig,
  Menu,
  MessageSquareText,
  Plus,
} from "lucide-vue-next";
import { computed, ref } from "vue";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import type { ActiveView, Conversation } from "@/types";

const props = defineProps<{
  conversations: Conversation[];
  selectedConversationId: number | null;
  activeView: ActiveView;
  memoryUpdated: boolean;
  busy: boolean;
  loading: boolean;
}>();

const emit = defineEmits<{
  create: [];
  select: [conversationId: number];
  openChat: [];
  openKnowledge: [];
  openMemory: [];
}>();

function conversationTitle(conversation: Conversation): string {
  return conversation.title ?? "未命名会话";
}

const mobileNavigationOpen = ref(false);
const mobileTitle = computed(() => {
  if (props.activeView === "knowledge") return "知识库";
  if (props.activeView === "memory") return "长期记忆";
  const conversation = props.conversations.find((item) => item.id === props.selectedConversationId);
  return conversation === undefined ? "聊天" : conversationTitle(conversation);
});

function closeMobileNavigation(): void {
  mobileNavigationOpen.value = false;
}

function createFromMobile(): void {
  emit("create");
  closeMobileNavigation();
}

function selectFromMobile(conversationId: number): void {
  emit("select", conversationId);
  closeMobileNavigation();
}

function openChatFromMobile(): void {
  emit("openChat");
  closeMobileNavigation();
}

function openKnowledgeFromMobile(): void {
  emit("openKnowledge");
  closeMobileNavigation();
}

function openMemoryFromMobile(): void {
  emit("openMemory");
  closeMobileNavigation();
}
</script>

<template>
  <div class="flex h-14 shrink-0 items-center justify-between border-b border-border bg-sidebar px-4 md:hidden">
    <div class="min-w-0">
      <p class="font-mono text-[9px] font-medium tracking-[0.14em] text-muted-light">
        LANGLEY
      </p>
      <p class="truncate text-sm font-semibold text-foreground">
        {{ mobileTitle }}
      </p>
    </div>
    <Sheet v-model:open="mobileNavigationOpen">
      <SheetTrigger as-child>
        <Button
          variant="ghost"
          size="icon"
          aria-label="打开导航"
        >
          <Menu
            :size="18"
            :stroke-width="1.8"
            aria-hidden="true"
          />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="left"
        class="w-[min(20rem,calc(100vw-2.5rem))] gap-0 border-strong-border bg-sidebar p-0 text-foreground"
      >
        <SheetHeader class="border-b border-border px-5 py-5 pr-14 text-left">
          <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
            LANGLEY
          </p>
          <SheetTitle class="mt-1 text-base font-semibold text-foreground">
            导航
          </SheetTitle>
          <SheetDescription class="mt-1 text-sm text-muted-foreground">
            访问聊天、知识库、长期记忆和最近会话。
          </SheetDescription>
        </SheetHeader>
        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto px-3 py-4">
          <Button
            class="w-full justify-start"
            :disabled="busy"
            @click="createFromMobile"
          >
            <Plus
              :size="15"
              :stroke-width="1.8"
              aria-hidden="true"
            />
            新建会话
          </Button>
          <p class="mb-2 mt-6 px-2 font-mono text-[10px] font-medium tracking-[0.15em] text-muted-light">
            WORKSPACE
          </p>
          <nav
            class="flex flex-col gap-0.5"
            aria-label="工作区"
          >
            <button
              type="button"
              class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              :class="activeView === 'chat' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
              :aria-current="activeView === 'chat' ? 'page' : undefined"
              @click="openChatFromMobile"
            >
              <MessageSquareText
                :size="15"
                :stroke-width="1.7"
                aria-hidden="true"
              />
              聊天
            </button>
            <button
              type="button"
              class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              :class="activeView === 'knowledge' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
              :aria-current="activeView === 'knowledge' ? 'page' : undefined"
              @click="openKnowledgeFromMobile"
            >
              <LibraryBig
                :size="15"
                :stroke-width="1.7"
                aria-hidden="true"
              />
              知识库
            </button>
            <button
              type="button"
              class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              :class="activeView === 'memory' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
              :aria-current="activeView === 'memory' ? 'page' : undefined"
              @click="openMemoryFromMobile"
            >
              <Brain
                :size="15"
                :stroke-width="1.7"
                aria-hidden="true"
              />
              <span>记忆</span>
              <span
                v-if="memoryUpdated"
                aria-label="记忆有更新"
                class="ml-auto size-1.5 rounded-full bg-primary"
              />
            </button>
          </nav>
          <div class="mb-2 mt-6 flex items-center justify-between gap-2 px-2">
            <p class="font-mono text-[10px] font-medium tracking-[0.15em] text-muted-light">
              RECENT
            </p>
            <span class="font-mono text-[10px] tabular-nums text-muted-light">{{ conversations.length }}</span>
          </div>
          <nav
            class="flex flex-col gap-0.5"
            aria-label="最近会话"
          >
            <button
              v-for="conversation in conversations"
              :key="conversation.id"
              type="button"
              class="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              :class="activeView === 'chat' && conversation.id === selectedConversationId ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
              :aria-current="activeView === 'chat' && conversation.id === selectedConversationId ? 'page' : undefined"
              @click="selectFromMobile(conversation.id)"
            >
              <MessageSquareText
                :size="14"
                :stroke-width="1.6"
                aria-hidden="true"
                class="shrink-0 opacity-70"
              />
              <span class="break-words">{{ conversationTitle(conversation) }}</span>
            </button>
            <p
              v-if="conversations.length === 0 && !loading"
              class="px-2 py-3 text-sm leading-6 text-muted-foreground"
            >
              新建一个会话后即可开始。
            </p>
          </nav>
        </div>
      </SheetContent>
    </Sheet>
  </div>

  <aside class="app-sidebar hidden w-[clamp(11rem,24vw,17rem)] shrink-0 flex-col border-r border-border bg-sidebar px-3 py-4 sm:px-4 sm:py-5 md:flex">
    <div class="flex items-center gap-2.5 px-1">
      <span class="flex size-7 shrink-0 items-center justify-center rounded-md border border-strong-border bg-surface text-primary">
        <BookOpenText
          :size="15"
          :stroke-width="1.7"
          aria-hidden="true"
        />
      </span>
      <div class="min-w-0">
        <p class="truncate text-sm font-semibold tracking-[-0.01em] text-foreground">
          Langley
        </p>
        <p class="truncate font-mono text-[9px] tracking-[0.14em] text-muted-light">
          PERSONAL KNOWLEDGE
        </p>
      </div>
    </div>

    <Button
      class="mt-6 w-full justify-start"
      :disabled="busy"
      @click="emit('create')"
    >
      <Plus
        :size="15"
        :stroke-width="1.8"
        aria-hidden="true"
      />
      新建会话
    </Button>

    <p class="mb-2 mt-7 px-2 font-mono text-[10px] font-medium tracking-[0.15em] text-muted-light">
      WORKSPACE
    </p>
    <nav
      class="space-y-0.5"
      aria-label="工作区"
    >
      <button
        type="button"
        class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        :class="activeView === 'chat' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
        :aria-current="activeView === 'chat' ? 'page' : undefined"
        @click="emit('openChat')"
      >
        <MessageSquareText
          :size="15"
          :stroke-width="1.7"
          aria-hidden="true"
        />
        聊天
      </button>
      <button
        type="button"
        class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        :class="activeView === 'knowledge' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
        :aria-current="activeView === 'knowledge' ? 'page' : undefined"
        @click="emit('openKnowledge')"
      >
        <LibraryBig
          :size="15"
          :stroke-width="1.7"
          aria-hidden="true"
        />
        知识库
      </button>
      <button
        type="button"
        class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        :class="activeView === 'memory' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
        :aria-current="activeView === 'memory' ? 'page' : undefined"
        @click="emit('openMemory')"
      >
        <Brain
          :size="15"
          :stroke-width="1.7"
          aria-hidden="true"
        />
        <span>记忆</span>
        <span
          v-if="memoryUpdated"
          aria-label="记忆有更新"
          class="ml-auto size-1.5 rounded-full bg-primary"
        />
      </button>
    </nav>

    <div class="mb-2 mt-7 flex items-center justify-between gap-2 px-2">
      <p class="font-mono text-[10px] font-medium tracking-[0.15em] text-muted-light">
        RECENT
      </p>
      <span class="font-mono text-[10px] tabular-nums text-muted-light">{{ conversations.length }}</span>
    </div>
    <nav
      class="min-h-0 flex-1 space-y-0.5 overflow-y-auto"
      aria-label="会话列表"
    >
      <button
        v-for="conversation in conversations"
        :key="conversation.id"
        type="button"
        class="group flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        :class="activeView === 'chat' && conversation.id === selectedConversationId ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
        :aria-current="activeView === 'chat' && conversation.id === selectedConversationId ? 'page' : undefined"
        @click="emit('select', conversation.id)"
      >
        <MessageSquareText
          :size="14"
          :stroke-width="1.6"
          aria-hidden="true"
          class="shrink-0 opacity-70"
        />
        <span class="truncate">{{ conversationTitle(conversation) }}</span>
      </button>
      <p
        v-if="conversations.length === 0 && !loading"
        class="px-2 py-3 text-sm leading-6 text-muted-foreground"
      >
        新建一个会话后即可开始。
      </p>
    </nav>
  </aside>
</template>
