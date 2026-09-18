<script setup lang="ts">
import { ArrowLeft, ChevronRight, FileText, Folder, RefreshCw } from "lucide-vue-next";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import type { Conversation, WorkspaceChanges } from "@/types";

const props = defineProps<{
  open: boolean;
  navigationBusy?: boolean;
  conversationId: number | null;
  conversations: Conversation[];
  workspaceId: number | null;
  changes: WorkspaceChanges | null;
}>();
const emit = defineEmits<{
  open: [conversationId: number];
  "update:open": [open: boolean];
  restoreFocus: [];
}>();
interface Workspace { id: number; name: string }
interface Entry { name: string; type: string }
class WorkspaceRequestError extends Error {
  constructor(readonly code: string | undefined) {
    super("Workspace request failed");
  }
}
const workspaces = ref<Workspace[]>([]);
const name = ref("");
const busy = ref(false);
const loadingFiles = ref(false);
const error = ref("");
const workspaceListError = ref("");
const unsupportedPreviewPath = ref<string | null>(null);
const notice = ref("");
const directory = ref("");
const entries = ref<Entry[]>([]);
const preview = ref<{ path: string; content: string; end_line: number; total_lines: number } | null>(null);
const folderInput = ref<HTMLInputElement | null>(null);
let scopeRevision = 0;
let fileRevision = 0;
let workspaceListRevision = 0;

async function json(response: Response) {
  const payload = await response.json();
  if (!response.ok) throw new WorkspaceRequestError(typeof payload?.detail?.code === "string" ? payload.detail.code : undefined);
  return payload;
}
async function loadWorkspaces() {
  const current = ++workspaceListRevision;
  try {
    const result = await json(await fetch("/api/workspaces"));
    if (current === workspaceListRevision) {
      workspaces.value = result;
      workspaceListError.value = "";
    }
  } catch {
    if (current === workspaceListRevision) workspaceListError.value = "工作区列表暂时不可用，请重新打开面板重试。";
  }
}
async function create(selectedFiles?: FileList | null) {
  if (busy.value || props.navigationBusy || (selectedFiles && selectedFiles.length === 0)) return;
  const current = scopeRevision;
  busy.value = true;
  error.value = "";
  notice.value = "";
  try {
    let response: Response;
    if (selectedFiles) {
      const body = new FormData();
      body.append("name", name.value.trim() || selectedFiles[0]?.webkitRelativePath.split("/")[0] || "导入的工作区");
      for (const file of Array.from(selectedFiles)) {
        const relative = file.webkitRelativePath.split("/").slice(1).join("/");
        body.append("files", file, relative);
      }
      response = await fetch("/api/workspaces/import", { method: "POST", body });
    } else {
      response = await fetch("/api/workspaces", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.value.trim() || "新工作区" }),
      });
    }
    const result = await json(response);
    await loadWorkspaces();
    if (current !== scopeRevision) return;
    const skipped = Object.values(result.skipped as Record<string, number>).reduce((a, b) => a + b, 0);
    notice.value = selectedFiles ? `已复制导入，跳过 ${skipped} 项（缓存目录或敏感文件名）。原文件夹不受后续编辑影响。` : "工作区已创建。";
    emit("open", result.conversation_id);
    name.value = "";
  } catch {
    if (current === scopeRevision) {
      error.value = "未能确认操作结果。请先刷新工作区列表，确认是否已创建，再决定是否重试。";
      await loadWorkspaces();
    }
  } finally {
    busy.value = false;
    if (folderInput.value) folderInput.value.value = "";
  }
}
async function openWorkspace(id: number, newChat = false) {
  if (busy.value || props.navigationBusy) return;
  const current = scopeRevision;
  busy.value = true;
  error.value = "";
  try {
    const existing = newChat ? undefined : props.conversations.find(c => c.workspace_id === id);
    const conversation = existing ?? await json(await fetch("/api/conversations", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_id: id }),
    }));
    if (current === scopeRevision) emit("open", conversation.id);
  } catch {
    if (current === scopeRevision) error.value = "未能确认会话是否打开，请先刷新会话列表后再试。";
  } finally { busy.value = false; }
}
async function files(path = "", isPreview = false, offset = 1) {
  const id = props.workspaceId;
  const scope = scopeRevision;
  const current = ++fileRevision;
  unsupportedPreviewPath.value = null;
  if (isPreview) preview.value = null;
  if (id === null) return;
  error.value = "";
  loadingFiles.value = true;
  try {
    const query = new URLSearchParams({ path, preview: String(isPreview), offset: String(offset) });
    const result = await json(await fetch(`/api/workspaces/${id}/files?${query}`));
    if (current !== fileRevision || scope !== scopeRevision || id !== props.workspaceId) return;
    if (isPreview) preview.value = result;
    else { entries.value = result.entries; directory.value = path; preview.value = null; }
  } catch (failure) {
    if (current !== fileRevision || scope !== scopeRevision) return;
    if (isPreview && failure instanceof WorkspaceRequestError && failure.code === "TEXT_FILE_UNSUPPORTED") {
      unsupportedPreviewPath.value = path;
    } else {
      error.value = "文件暂时无法读取，可能已变更或不支持文本预览。请刷新文件后重试。";
    }
  } finally {
    if (current === fileRevision && scope === scopeRevision) loadingFiles.value = false;
  }
}
function parentDirectory() {
  if (directory.value) void files(directory.value.split("/").slice(0, -1).join("/"));
}
function importSelected(event: Event) {
  const selectedFiles = (event.target as HTMLInputElement).files;
  if (selectedFiles?.length) void create(selectedFiles);
}
function restoreFocus(event: Event) {
  event.preventDefault();
  emit("restoreFocus");
}
watch(() => [props.conversationId, props.workspaceId], () => {
  scopeRevision++;
  fileRevision++;
  entries.value = [];
  preview.value = null;
  unsupportedPreviewPath.value = null;
  directory.value = "";
  error.value = "";
  loadingFiles.value = false;
  if (props.open && props.workspaceId !== null) void files();
});
watch(() => props.changes, () => {
  if (props.open && props.workspaceId !== null) void files(directory.value);
});
watch(() => props.open, (open) => {
  if (open) {
    void loadWorkspaces();
    if (props.workspaceId !== null) void files(directory.value);
  }
});
onMounted(() => {
  void loadWorkspaces();
  if (props.open && props.workspaceId !== null) void files();
});
onBeforeUnmount(() => { scopeRevision++; fileRevision++; workspaceListRevision++; });
</script>

<template>
  <Sheet
    :open="open"
    @update:open="emit('update:open', $event)"
  >
    <SheetContent
      side="right"
      class="w-full gap-0 overflow-y-auto bg-surface p-5 text-sm shadow-none sm:max-w-md sm:p-6"
      @close-auto-focus="restoreFocus"
    >
      <SheetHeader class="p-0 pr-8">
        <SheetTitle class="text-xl font-semibold">
          工作区
        </SheetTitle>
        <SheetDescription class="mt-1 leading-[1.6]">
          {{ workspaceId !== null ? "当前会话正在此工作区中运行。" : "选择或创建工作区后，将进入该工作区的会话。" }}
        </SheetDescription>
      </SheetHeader>

      <p
        v-if="notice"
        role="status"
        class="mt-4 text-[13px] leading-6 text-success-foreground"
      >
        {{ notice }}
      </p>
      <div
        v-if="error || workspaceListError"
        class="mt-4 text-[13px] leading-6"
      >
        <p
          v-if="error"
          role="alert"
          class="text-warning-foreground"
        >
          {{ error }}
        </p>
        <p
          v-if="workspaceListError"
          role="alert"
          class="text-warning-foreground"
        >
          {{ workspaceListError }}
        </p>
        <Button
          variant="outline"
          size="small"
          class="mt-2"
          @click="loadWorkspaces"
        >
          刷新工作区列表
        </Button>
      </div>

      <div class="mt-6 border-b border-border pb-5">
        <label class="flex flex-col gap-2 text-sm font-medium text-body">当前工作区
          <select
            :value="workspaceId ?? ''"
            :disabled="busy || navigationBusy"
            class="w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-sm font-normal text-foreground focus-visible:outline focus-visible:outline-ring"
            @change="openWorkspace(Number(($event.target as HTMLSelectElement).value))"
          >
            <option
              disabled
              value=""
            >选择工作区</option>
            <option
              v-for="workspace in workspaces"
              :key="workspace.id"
              :value="workspace.id"
            >{{ workspace.name }}</option>
          </select>
        </label>
        <Button
          v-if="workspaceId !== null"
          variant="ghost"
          size="small"
          class="mt-2"
          :disabled="busy || navigationBusy"
          @click="openWorkspace(workspaceId, true)"
        >
          在此工作区新建会话
        </Button>
      </div>

      <template v-if="workspaceId !== null">
        <section
          aria-label="文件"
          class="py-5"
          :aria-busy="loadingFiles"
        >
          <div class="flex items-center justify-between gap-3">
            <h2 class="text-[17px] font-semibold text-foreground">
              文件
            </h2>
            <Button
              variant="ghost"
              size="small"
              :disabled="loadingFiles"
              @click="files(directory)"
            >
              <RefreshCw
                :size="14"
                aria-hidden="true"
              />
              刷新文件
            </Button>
          </div>
          <div class="mt-2 flex items-center gap-2">
            <Button
              variant="ghost"
              size="small"
              :disabled="!directory"
              @click="parentDirectory"
            >
              <ArrowLeft
                :size="14"
                aria-hidden="true"
              />
              上一级
            </Button>
            <Button
              variant="ghost"
              size="small"
              :disabled="!directory"
              @click="files()"
            >
              根目录
            </Button>
          </div>
          <p class="my-3 break-all font-mono text-[13px] leading-5 text-muted-foreground">
            /{{ directory }}
          </p>
          <p
            v-if="loadingFiles"
            role="status"
            class="py-2 text-[13px] text-muted-foreground"
          >
            正在读取文件…
          </p>
          <ul class="flex flex-col gap-0.5">
            <li
              v-for="entry in entries"
              :key="entry.name"
            >
              <button
                type="button"
                class="file-entry flex w-full items-center gap-2.5 rounded-sm px-2 py-2 text-left text-foreground hover:bg-subtle focus-visible:outline focus-visible:outline-ring disabled:text-muted-light"
                :class="{ 'bg-subtle': (preview?.path ?? unsupportedPreviewPath) === [directory, entry.name].filter(Boolean).join('/') }"
                :disabled="entry.type === 'symlink'"
                :aria-current="(preview?.path ?? unsupportedPreviewPath) === [directory, entry.name].filter(Boolean).join('/') ? 'true' : undefined"
                @click="files([directory, entry.name].filter(Boolean).join('/'), entry.type !== 'directory')"
              >
                <component
                  :is="entry.type === 'directory' ? Folder : FileText"
                  :size="16"
                  class="shrink-0 text-muted-foreground"
                  aria-hidden="true"
                />
                <span class="min-w-0 flex-1 break-all">{{ entry.name }}</span>
                <span
                  v-if="entry.type === 'symlink'"
                  class="text-[13px]"
                >符号链接 · 不可打开</span>
                <ChevronRight
                  v-if="entry.type === 'directory'"
                  :size="14"
                  class="shrink-0 text-muted-foreground"
                  aria-hidden="true"
                />
              </button>
            </li>
          </ul>
          <p
            v-if="!loadingFiles && entries.length === 0"
            class="py-3 text-sm text-muted-foreground"
          >
            当前目录为空。
          </p>
        </section>

        <section
          v-if="unsupportedPreviewPath !== null"
          aria-label="文件预览"
          class="border-t border-border py-5"
        >
          <h2 class="text-base font-semibold text-foreground">
            文件预览
          </h2>
          <p class="mt-2 break-all font-mono text-[13px] text-muted-foreground">
            {{ unsupportedPreviewPath }}
          </p>
          <p
            role="status"
            class="mt-3 text-sm leading-6 text-muted-foreground"
          >
            此文件不支持文本预览。二进制文件仍保留在工作区中。
          </p>
        </section>
        <section
          v-else-if="preview"
          aria-label="文件预览"
          class="border-t border-border py-5"
        >
          <h2 class="text-base font-semibold text-foreground">
            文件预览
          </h2>
          <p class="mt-2 break-all font-mono text-[13px] leading-5 text-muted-foreground">
            {{ preview.path }}
          </p>
          <pre class="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-sm bg-subtle p-3 font-mono text-[13px] leading-[1.6] text-body">{{ preview.content }}</pre>
          <div class="mt-3 flex items-center justify-between gap-3">
            <p class="text-[13px] text-muted-foreground">
              已读至第 {{ preview.end_line }} 行 · 共 {{ preview.total_lines }} 行
            </p>
            <Button
              v-if="preview.end_line < preview.total_lines"
              size="small"
              variant="outline"
              :disabled="loadingFiles"
              @click="files(preview.path, true, preview.end_line + 1)"
            >
              下一页
            </Button>
          </div>
        </section>

        <section
          v-if="changes"
          aria-label="本次文件变化"
          class="border-t border-border py-5"
        >
          <h2 class="text-base font-semibold text-foreground">
            本次文件变化
          </h2>
          <p class="mt-2 text-[13px] text-muted-foreground">
            新增 {{ changes.added.length }} · 修改 {{ changes.modified.length }} · 删除 {{ changes.deleted.length }}
          </p>
          <p
            v-if="!changes.complete"
            class="mt-2 text-[13px] text-warning-foreground"
          >
            扫描达到上限，结果可能不完整。
          </p>
          <ul class="mt-3 flex flex-col gap-1.5 text-sm">
            <li
              v-for="path in changes.added"
              :key="`+${path}`"
              class="break-all text-success-foreground"
            >
              <span class="mr-2">新增</span>{{ path }}
            </li>
            <li
              v-for="path in changes.modified"
              :key="`~${path}`"
              class="break-all text-body"
            >
              <span class="mr-2">修改</span>{{ path }}
            </li>
            <li
              v-for="path in changes.deleted"
              :key="`-${path}`"
              class="break-all text-danger-foreground"
            >
              <span class="mr-2">删除</span>{{ path }}
            </li>
          </ul>
          <p
            v-if="changes.complete && !changes.added.length && !changes.modified.length && !changes.deleted.length"
            class="mt-3 text-sm text-body"
          >
            本次运行未检测到文件变化。
          </p>
        </section>
      </template>
      <p
        v-else
        class="py-7 text-sm leading-6 text-muted-foreground"
      >
        打开工作区后，在这里浏览文件和查看运行变化。
      </p>

      <details class="border-t border-border py-4 text-sm">
        <summary class="cursor-pointer font-medium text-body focus-visible:outline focus-visible:outline-ring">
          工作区管理
        </summary>
        <div class="mt-4 flex flex-col gap-3">
          <label class="flex flex-col gap-2 font-medium text-body">工作区名称
            <input
              v-model="name"
              maxlength="255"
              class="workspace-name w-full rounded-sm border border-strong-border bg-surface px-3 py-2 text-foreground focus-visible:outline focus-visible:outline-ring"
              placeholder="新工作区"
              :disabled="busy || navigationBusy"
            >
          </label>
          <div class="flex flex-wrap gap-2">
            <Button
              size="small"
              variant="outline"
              :disabled="busy || navigationBusy"
              @click="create()"
            >
              新建工作区
            </Button>
            <Button
              size="small"
              variant="outline"
              :disabled="busy || navigationBusy"
              @click="folderInput?.click()"
            >
              导入文件夹
            </Button>
          </div>
          <input
            ref="folderInput"
            type="file"
            webkitdirectory
            multiple
            class="hidden"
            aria-label="选择导入文件夹"
            :disabled="busy || navigationBusy"
            @change="importSelected"
          >
          <p class="text-[13px] leading-6 text-muted-foreground">
            导入后使用托管副本，原文件夹不会被后续修改。
          </p>
        </div>
      </details>
      <details class="border-t border-border py-4 text-[13px] leading-6 text-muted-foreground">
        <summary class="cursor-pointer focus-visible:outline focus-visible:outline-ring">
          文件与运行说明
        </summary>
        <p class="mt-2">
          失败或停止不会撤销文件变化。重试会基于当前工作区文件重新规划。
        </p>
        <p class="mt-2">
          变化摘要临时保留，服务重启后不可恢复，不是永久变更记录。
        </p>
      </details>
    </SheetContent>
  </Sheet>
</template>

<style scoped>
button[data-slot="button"] {
  font-size: 14px;
}

.file-entry,
.workspace-name {
  font-size: 14px;
  font-weight: 400;
}
</style>
