<script setup lang="ts">
import {
  Activity,
  BookMarked,
  BookOpenText,
  Brain,
  Folder,
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

const props = withDefaults(defineProps<{
  conversations: Conversation[];
  selectedConversationId: number | null;
  activeView: ActiveView;
  memoryUpdated: boolean;
  busy: boolean;
  loading: boolean;
  developerEnabled?: boolean;
}>(), {
  developerEnabled: false,
});

const emit = defineEmits<{
  create: [];
  select: [conversationId: number];
  openChat: [];
  openKnowledge: [];
  openMemory: [];
  openSkills: [];
  openObservatory: [];
}>();

function conversationTitle(conversation: Conversation): string {
  return conversation.title ?? "未命名会话";
}

const conversationGroups = computed(() => [
  { title: "工作区会话", workspace: true, conversations: props.conversations.filter(conversation => conversation.workspace_id != null) },
  { title: "最近会话", workspace: false, conversations: props.conversations.filter(conversation => conversation.workspace_id == null) },
]);

const mobileNavigationOpen = ref(false);
const mobileTitle = computed(() => {
  if (props.activeView === "knowledge") return "知识库";
  if (props.activeView === "memory") return "长期记忆";
  if (props.activeView === "skills") return "技能";
  if (props.activeView === "observatory") return "运行观测台";
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

function openSkillsFromMobile(): void {
  emit("openSkills");
  closeMobileNavigation();
}

function openObservatoryFromMobile(): void {
  emit("openObservatory");
  closeMobileNavigation();
}
</script>

<template>
  <div class="flex h-14 shrink-0 items-center justify-between border-b border-border bg-sidebar px-4 md:hidden">
    <div class="min-w-0">
      <p class="text-xs font-normal text-muted-light">
        Langley
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
          <p class="text-xs font-normal text-muted-light">
            Langley
          </p>
          <SheetTitle class="mt-1 text-base font-semibold text-foreground">
            导航
          </SheetTitle>
          <SheetDescription class="mt-1 text-sm text-muted-foreground">
            访问聊天、知识库、长期记忆、技能和最近会话。
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
          <p class="mb-2 mt-6 px-2 text-xs font-normal text-muted-light">
            导航
          </p>
          <nav
            class="flex flex-col gap-0.5"
            aria-label="导航"
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
            <button
              type="button"
              class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              :class="activeView === 'skills' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
              :aria-current="activeView === 'skills' ? 'page' : undefined"
              @click="openSkillsFromMobile"
            >
              <BookMarked
                :size="15"
                :stroke-width="1.7"
                aria-hidden="true"
              />
              技能
            </button>
          </nav>
          <template v-if="developerEnabled">
            <p class="mb-2 mt-6 px-2 text-xs font-normal text-muted-light">
              开发者
            </p>
            <nav
              aria-label="开发者"
              class="flex flex-col gap-0.5"
            >
              <button
                type="button"
                class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                :class="activeView === 'observatory' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
                :aria-current="activeView === 'observatory' ? 'page' : undefined"
                @click="openObservatoryFromMobile"
              >
                <Activity
                  :size="15"
                  :stroke-width="1.7"
                  aria-hidden="true"
                />
                运行观测
              </button>
            </nav>
          </template>
          <div class="mt-6">
            <nav
              v-for="group in conversationGroups"
              :key="group.title"
              :aria-label="group.title"
              class="mb-6 flex flex-col gap-0.5"
            >
              <div class="mb-2 flex items-center justify-between gap-2 px-2">
                <h2 class="text-xs font-normal text-muted-light">
                  {{ group.title }}
                </h2>
                <span class="font-mono text-[10px] tabular-nums text-muted-light">{{ group.conversations.length }}</span>
              </div>
              <button
                v-for="conversation in group.conversations"
                :key="conversation.id"
                type="button"
                class="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                :class="activeView === 'chat' && conversation.id === selectedConversationId ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
                :aria-current="activeView === 'chat' && conversation.id === selectedConversationId ? 'page' : undefined"
                @click="selectFromMobile(conversation.id)"
              >
                <component
                  :is="group.workspace ? Folder : MessageSquareText"
                  :size="14"
                  :stroke-width="1.6"
                  aria-hidden="true"
                  class="shrink-0"
                  :class="group.workspace ? 'text-muted-light' : 'opacity-70'"
                />
                <span class="break-words">{{ conversationTitle(conversation) }}</span>
              </button>
            </nav>
            <p
              v-if="conversations.length === 0 && !loading"
              class="px-2 py-3 text-sm leading-6 text-muted-foreground"
            >
              新建一个会话后即可开始。
            </p>
          </div>
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
        <p class="truncate text-xs font-normal text-muted-light">
          Personal knowledge
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

    <p class="mb-2 mt-7 px-2 text-xs font-normal text-muted-light">
      导航
    </p>
    <nav
      class="space-y-0.5"
      aria-label="导航"
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
      <button
        type="button"
        class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        :class="activeView === 'skills' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
        :aria-current="activeView === 'skills' ? 'page' : undefined"
        @click="emit('openSkills')"
      >
        <BookMarked
          :size="15"
          :stroke-width="1.7"
          aria-hidden="true"
        />
        技能
      </button>
    </nav>

    <template v-if="developerEnabled">
      <p class="mb-2 mt-7 px-2 text-xs font-normal text-muted-light">
        开发者
      </p>
      <nav
        aria-label="开发者"
        class="space-y-0.5"
      >
        <button
          type="button"
          class="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          :class="activeView === 'observatory' ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
          :aria-current="activeView === 'observatory' ? 'page' : undefined"
          @click="emit('openObservatory')"
        >
          <Activity
            :size="15"
            :stroke-width="1.7"
            aria-hidden="true"
          />
          运行观测
        </button>
      </nav>
    </template>

    <div class="mt-7 min-h-0 flex-1 overflow-y-auto">
      <nav
        v-for="group in conversationGroups"
        :key="group.title"
        :aria-label="group.title"
        class="mb-6 flex flex-col gap-0.5"
      >
        <div class="mb-2 flex items-center justify-between gap-2 px-2">
          <h2 class="text-xs font-normal text-muted-light">
            {{ group.title }}
          </h2>
          <span class="font-mono text-[10px] tabular-nums text-muted-light">{{ group.conversations.length }}</span>
        </div>
        <button
          v-for="conversation in group.conversations"
          :key="conversation.id"
          type="button"
          class="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          :class="activeView === 'chat' && conversation.id === selectedConversationId ? 'bg-surface text-foreground shadow-[inset_2px_0_0_var(--primary)]' : 'text-muted-foreground hover:bg-subtle hover:text-foreground'"
          :aria-current="activeView === 'chat' && conversation.id === selectedConversationId ? 'page' : undefined"
          @click="emit('select', conversation.id)"
        >
          <component
            :is="group.workspace ? Folder : MessageSquareText"
            :size="14"
            :stroke-width="1.6"
            aria-hidden="true"
            class="shrink-0"
            :class="group.workspace ? 'text-muted-light' : 'opacity-70'"
          />
          <span class="truncate">{{ conversationTitle(conversation) }}</span>
        </button>
      </nav>
      <p
        v-if="conversations.length === 0 && !loading"
        class="px-2 py-3 text-sm leading-6 text-muted-foreground"
      >
        新建一个会话后即可开始。
      </p>
    </div>
  </aside>
</template>

<style scoped>
button[aria-current="page"] {
  font-weight: 500;
}
</style>
