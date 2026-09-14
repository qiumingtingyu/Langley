<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { Button } from "@/components/ui/button";
import type { Conversation, WorkspaceChanges } from "@/types";

const props = defineProps<{
  conversations: Conversation[];
  workspaceId: number | null;
  changes: WorkspaceChanges | null;
}>();
const emit = defineEmits<{ open: [conversationId: number] }>();
interface Workspace { id: number; name: string }
interface Entry { name: string; type: string }
const workspaces = ref<Workspace[]>([]);
const name = ref("");
const busy = ref(false);
const error = ref("");
const notice = ref("");
const directory = ref("");
const entries = ref<Entry[]>([]);
const preview = ref<{ path: string; content: string; end_line: number; total_lines: number } | null>(null);
const folderInput = ref<HTMLInputElement | null>(null);
let revision = 0;

async function json(response: Response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail?.code ?? "请求失败");
  return payload;
}
async function loadWorkspaces() {
  try { workspaces.value = await json(await fetch("/api/workspaces")); }
  catch (e) { error.value = String(e); }
}
async function create(files?: FileList | null) {
  const current = revision;
  busy.value = true;
  error.value = "";
  try {
    let response: Response;
    if (files) {
      const body = new FormData();
      body.append("name", name.value.trim() || files[0]?.webkitRelativePath.split("/")[0] || "导入的工作区");
      for (const file of Array.from(files)) {
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
    if (current !== revision) return;
    const skipped = Object.values(result.skipped as Record<string, number>).reduce((a, b) => a + b, 0);
    notice.value = files ? `已复制导入，跳过 ${skipped} 项（缓存目录或敏感文件名）。原文件夹不受后续编辑影响。` : "工作区已创建。";
    emit("open", result.conversation_id);
    name.value = "";
  } catch (e) { if (current === revision) error.value = String(e); }
  finally { busy.value = false; if (folderInput.value) folderInput.value.value = ""; }
}
async function openWorkspace(id: number, newChat = false) {
  const current = ++revision;
  busy.value = true;
  error.value = "";
  try {
    const existing = newChat ? undefined : props.conversations.find(c => c.workspace_id === id);
    const conversation = existing ?? await json(await fetch("/api/conversations", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_id: id }),
    }));
    if (current === revision) emit("open", conversation.id);
  } catch (e) { if (current === revision) error.value = String(e); }
  finally { busy.value = false; }
}
async function files(path = "", isPreview = false, offset = 1) {
  const id = props.workspaceId;
  const current = ++revision;
  if (id === null) return;
  error.value = "";
  try {
    const query = new URLSearchParams({ path, preview: String(isPreview), offset: String(offset) });
    const result = await json(await fetch(`/api/workspaces/${id}/files?${query}`));
    if (current !== revision || id !== props.workspaceId) return;
    if (isPreview) preview.value = result;
    else { entries.value = result.entries; directory.value = path; preview.value = null; }
  } catch (e) { if (current === revision) error.value = String(e); }
}
function importSelected(event: Event) { void create((event.target as HTMLInputElement).files); }
watch(() => props.workspaceId, () => {
  revision++;
  entries.value = [];
  preview.value = null;
  directory.value = "";
  void files();
});
watch(() => props.changes, () => { if (props.workspaceId !== null) void files(directory.value); });
onMounted(() => { void loadWorkspaces(); if (props.workspaceId !== null) void files(); });
</script>

<template>
  <aside
    aria-label="工作区"
    class="flex max-h-60 shrink-0 flex-col gap-3 overflow-auto border-b border-border bg-surface p-3 text-sm md:max-h-none md:w-64 md:border-b-0 md:border-r"
  >
    <details :open="workspaceId !== null">
      <summary class="cursor-pointer font-semibold">
        工作区
      </summary>
      <div class="mt-3 flex flex-col gap-3">
        <label class="flex flex-col gap-1">工作区名称
          <input
            v-model="name"
            maxlength="255"
            class="rounded border border-border bg-canvas p-2"
            placeholder="新工作区"
            :disabled="busy"
          >
        </label>
        <div class="flex flex-wrap gap-2">
          <Button
            size="small"
            variant="outline"
            :disabled="busy"
            @click="create()"
          >
            新建工作区
          </Button>
          <Button
            size="small"
            variant="outline"
            :disabled="busy"
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
          @change="importSelected"
        >
        <label class="flex flex-col gap-1">打开工作区
          <select
            :value="workspaceId ?? ''"
            :disabled="busy"
            class="rounded border border-border bg-canvas p-2"
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
        <template v-if="workspaceId !== null">
          <Button
            variant="outline"
            size="small"
            :disabled="busy"
            @click="openWorkspace(workspaceId, true)"
          >
            在此工作区新建会话
          </Button>
          <p class="break-all font-mono text-xs">
            /{{ directory }}
          </p>
          <div class="flex gap-2">
            <Button
              variant="ghost"
              size="small"
              @click="files()"
            >
              根目录
            </Button>
            <Button
              variant="ghost"
              size="small"
              @click="files(directory)"
            >
              刷新文件
            </Button>
          </div>
          <ul class="flex flex-col gap-1">
            <li
              v-for="entry in entries"
              :key="entry.name"
            >
              <button
                class="w-full break-all text-left underline-offset-4 hover:underline"
                :disabled="entry.type === 'symlink'"
                @click="files([directory, entry.name].filter(Boolean).join('/'), entry.type !== 'directory')"
              >
                {{ entry.type === 'directory' ? '▸ ' : '' }}{{ entry.name }}
              </button>
            </li>
          </ul>
          <p
            v-if="entries.length === 0"
            class="text-muted-foreground"
          >
            当前目录为空。
          </p>
          <template v-if="preview">
            <p class="break-all font-mono text-xs">
              {{ preview.path }}
            </p>
            <pre class="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded bg-canvas p-2 text-xs">{{ preview.content }}</pre>
            <Button
              v-if="preview.end_line < preview.total_lines"
              size="small"
              variant="outline"
              @click="files(preview.path, true, preview.end_line + 1)"
            >
              下一页
            </Button>
          </template>
          <section
            v-if="changes"
            aria-label="本次文件变化"
            class="flex flex-col gap-1"
          >
            <h3 class="font-semibold">
              本次文件变化
            </h3>
            <p v-if="!changes.complete">
              扫描达到上限，以下结果不完整。
            </p>
            <p
              v-for="path in changes.added"
              :key="`+${path}`"
              class="break-all"
            >
              + {{ path }}
            </p>
            <p
              v-for="path in changes.modified"
              :key="`~${path}`"
              class="break-all"
            >
              ~ {{ path }}
            </p>
            <p
              v-for="path in changes.deleted"
              :key="`-${path}`"
              class="break-all"
            >
              − {{ path }}
            </p>
            <p v-if="changes.complete && !changes.added.length && !changes.modified.length && !changes.deleted.length">
              未检测到文件变化。
            </p>
          </section>
          <p class="text-xs text-muted-foreground">
            失败或停止不会撤销文件变化。变化摘要临时保留，服务重启后不可恢复。
          </p>
        </template>
      </div>
    </details>
    <p
      v-if="notice"
      role="status"
    >
      {{ notice }}
    </p>
    <p
      v-if="error"
      role="alert"
    >
      {{ error }}
    </p>
  </aside>
</template>
