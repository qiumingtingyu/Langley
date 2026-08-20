<script setup lang="ts">
import { onMounted, ref } from "vue";

import { Button } from "@/components/ui/button";

type Memory = { id: number; content: string; valid_until: string | null; source_message_id: number | null; created_at: string; updated_at: string };
type Source = { kind: "direct" } | { kind: "conversation"; conversation_title: string | null; conversation_deleted: boolean; context_messages: Array<{ id: number; role: string; content: string }> };

const emit = defineEmits<{ notice: [message: string]; }>();
const memories = ref<Memory[]>([]);
const enabled = ref(false);
const loading = ref(false);
const saving = ref(false);
const content = ref("");
const validUntil = ref("");
const editing = ref<number | null>(null);
const source = ref<Source | null>(null);
const sourceFor = ref<number | null>(null);
let loadRevision = 0;
let sourceRevision = 0;

function errorMessage(payload: unknown): string {
  const code = typeof payload === "object" && payload !== null && "detail" in payload ? (payload.detail as { code?: string }).code : undefined;
  if (code === "MEMORY_NOT_FOUND") return "这条长期记忆已不存在。";
  if (code === "MEMORY_SYNC_UNAVAILABLE") return "长期记忆同步暂时无法完成，请稍后重试。";
  if (code === "VALIDATION_ERROR") return "输入内容无效，请检查后重试。";
  return "操作失败，请稍后重试。";
}

async function request(path: string, init?: Parameters<typeof window.fetch>[1]): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error(errorMessage(await response.json()));
  return response;
}

async function load(): Promise<void> {
  const revision = ++loadRevision;
  loading.value = true;
  try {
    const [settingsResponse, memoriesResponse] = await Promise.all([request("/api/memory-settings"), request("/api/memories")]);
    const [settings, nextMemories] = await Promise.all([
      settingsResponse.json() as Promise<{ auto_memory_enabled: boolean }>,
      memoriesResponse.json() as Promise<Memory[]>,
    ]);
    if (revision !== loadRevision) return;
    enabled.value = settings.auto_memory_enabled;
    memories.value = nextMemories;
  } catch (error) { if (revision === loadRevision) emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { if (revision === loadRevision) loading.value = false; }
}

function absoluteValidUntil(): string | null { return validUntil.value ? new Date(validUntil.value).toISOString() : null; }
function resetForm(): void { content.value = ""; validUntil.value = ""; editing.value = null; }
async function save(): Promise<void> {
  if (!content.value.trim()) { emit("notice", "输入内容无效，请检查后重试。"); return; }
  saving.value = true;
  try {
    const path = editing.value === null ? "/api/memories" : `/api/memories/${editing.value}`;
    await request(path, { method: editing.value === null ? "POST" : "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: content.value.trim(), valid_until: absoluteValidUntil() }) });
    resetForm(); await load(); emit("notice", "长期记忆已更新。");
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { saving.value = false; }
}
async function toggle(): Promise<void> {
  const prior = enabled.value; saving.value = true;
  try { const response = await request("/api/memory-settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ auto_memory_enabled: !prior }) }); enabled.value = (await response.json() as { auto_memory_enabled: boolean }).auto_memory_enabled; }
  catch (error) { enabled.value = prior; emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { saving.value = false; }
}
function localDatetimeInput(value: string): string {
  const date = new Date(value);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 19);
}
function edit(memory: Memory): void { editing.value = memory.id; content.value = memory.content; validUntil.value = memory.valid_until ? localDatetimeInput(memory.valid_until) : ""; }
async function forget(memory: Memory): Promise<void> {
  if (!window.confirm("确定忘记这条长期记忆吗？")) return;
  try { await request(`/api/memories/${memory.id}`, { method: "DELETE" }); await load(); }
  catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
}
async function showSource(memory: Memory): Promise<void> {
  const revision = ++sourceRevision;
  sourceFor.value = memory.id;
  source.value = null;
  try {
    const next = await (await request(`/api/memories/${memory.id}/source`)).json() as Source;
    if (revision === sourceRevision && sourceFor.value === memory.id) source.value = next;
  } catch (error) {
    if (revision === sourceRevision && sourceFor.value === memory.id) emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。");
  }
}
defineExpose({ load });
onMounted(() => void load());
</script>

<template>
  <section class="mx-auto w-full max-w-3xl px-5 py-8 sm:px-8">
    <div class="mb-7 flex items-center justify-between rounded-lg border border-stone-200 bg-white p-4">
      <div>
        <h1 class="font-semibold text-slate-900">
          长期记忆
        </h1><p class="mt-1 text-sm text-slate-500">
          自动记忆：{{ enabled ? "开" : "关" }}
        </p>
      </div>
      <Button
        :disabled="saving"
        @click="toggle"
      >
        {{ saving ? "正在同步…" : enabled ? "关闭" : "开启" }}
      </Button>
    </div>
    <form
      class="mb-7 space-y-3 rounded-lg border border-stone-200 bg-white p-4"
      @submit.prevent="save"
    >
      <label class="block text-sm font-medium">{{ editing === null ? "新增记忆" : "修改记忆" }}<textarea
        v-model="content"
        class="mt-2 w-full rounded border border-stone-300 p-2"
        maxlength="1000"
      /></label>
      <label class="block text-sm">有效期（可选）<input
        v-model="validUntil"
        type="datetime-local"
        step="1"
        class="ml-2 rounded border border-stone-300 p-1"
      ></label>
      <Button
        type="submit"
        :disabled="saving"
      >
        {{ editing === null ? "添加" : "保存" }}
      </Button><Button
        v-if="editing !== null"
        type="button"
        variant="ghost"
        class="ml-2"
        @click="resetForm"
      >
        取消
      </Button>
    </form>
    <p
      v-if="loading"
      class="text-sm text-slate-500"
    >
      正在加载…
    </p>
    <div
      v-else-if="memories.length === 0"
      class="rounded-lg border border-dashed border-stone-300 p-8 text-center text-sm text-slate-500"
    >
      还没有长期记忆。<br>Langley 会在合适的对话中保存对未来有帮助的信息，你也可以手动添加。
    </div>
    <div
      v-else
      class="space-y-3"
    >
      <article
        v-for="memory in memories"
        :key="memory.id"
        class="rounded-lg border border-stone-200 bg-white p-4"
      >
        <p class="break-words whitespace-pre-wrap">
          {{ memory.content }}
        </p><p class="mt-2 text-xs text-slate-500">
          {{ memory.valid_until ? `有效期：${new Date(memory.valid_until).toLocaleString()}` : "有效期：长期有效" }}
        </p><div class="mt-3 flex gap-2">
          <Button
            size="small"
            variant="ghost"
            @click="showSource(memory)"
          >
            来源
          </Button><Button
            size="small"
            variant="ghost"
            @click="edit(memory)"
          >
            修改
          </Button><Button
            size="small"
            variant="ghost"
            @click="forget(memory)"
          >
            忘记
          </Button>
        </div><div
          v-if="sourceFor === memory.id && source"
          class="mt-3 rounded bg-stone-50 p-3 text-sm"
        >
          <p v-if="source.kind === 'direct'">
            这条记忆由你直接设置或修改。
          </p><template v-else>
            <p
              v-if="source.conversation_deleted"
              class="mb-2 text-slate-500"
            >
              原会话已删除。
            </p><p
              v-for="message in source.context_messages"
              :key="message.id"
              :class="message.id === memory.source_message_id ? 'font-semibold text-slate-900' : 'text-slate-600'"
            >
              {{ message.content }}
            </p>
          </template>
        </div>
      </article>
    </div>
  </section>
</template>
