<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { Button } from "@/components/ui/button";

type KnowledgeBase = { id: number; name: string; created_at: string };
type DocumentSource = { document_version_id: number; filename: string; media_type: string; size_bytes: number; sha256: string; created_at: string };
type Document = { id: number; name: string; created_at: string; source: DocumentSource };
type VerifyResult = { document_version_id: number; verified: boolean; verified_at: string };

const emit = defineEmits<{ notice: [message: string] }>();
const knowledgeBases = ref<KnowledgeBase[]>([]);
const documents = ref<Document[]>([]);
const selectedKnowledgeBaseId = ref<number | null>(null);
const selectedDocumentId = ref<number | null>(null);
const knowledgeBaseName = ref("");
const selectedFile = ref<File | null>(null);
const documentName = ref("");
const loading = ref(false);
const creating = ref(false);
const uploading = ref(false);
const verifying = ref(false);
const verifiedVersionId = ref<number | null>(null);
const verifiedAt = ref<string | null>(null);

const selectedDocument = computed(() => documents.value.find((item) => item.id === selectedDocumentId.value) ?? null);

function errorMessage(payload: unknown): string {
  const code = typeof payload === "object" && payload !== null && "detail" in payload ? (payload.detail as { code?: string }).code : undefined;
  if (code === "KNOWLEDGE_BASE_NOT_FOUND" || code === "DOCUMENT_VERSION_NOT_FOUND") return "该知识库或文档已不存在，请刷新后重试。";
  if (code === "UPLOAD_TOO_LARGE") return "Markdown 文件不能超过 5 MiB。";
  if (code === "EMPTY_SOURCE") return "文件内容不能为空。";
  if (code === "INVALID_SOURCE_ENCODING") return "文件必须是 UTF-8 编码的 Markdown。";
  if (code === "SOURCE_MISSING") return "原始文件缺失，无法验证。";
  if (code === "SOURCE_INTEGRITY_MISMATCH") return "原始文件与保存时的完整性信息不一致。";
  if (code === "SOURCE_STORAGE_FAILED") return "原始文件存储暂时不可用，请稍后重试。";
  if (code === "VALIDATION_ERROR") return "输入内容无效，请检查后重试。";
  return "操作失败，请稍后重试。";
}

class KnowledgeRequestError extends Error { constructor(readonly code: string | undefined, message: string) { super(message); } }

async function request(path: string, init?: Parameters<typeof window.fetch>[1]): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) { const payload = await response.json(); const code = typeof payload === "object" && payload !== null && "detail" in payload ? (payload.detail as { code?: string }).code : undefined; throw new KnowledgeRequestError(code, errorMessage(payload)); }
  return response;
}

async function loadDocuments(knowledgeBaseId: number, preferredDocumentId?: number): Promise<void> {
  const response = await request(`/api/knowledge-bases/${knowledgeBaseId}/documents`);
  const nextDocuments = await response.json() as Document[];
  if (selectedKnowledgeBaseId.value !== knowledgeBaseId) return;
  documents.value = nextDocuments;
  selectedDocumentId.value = preferredDocumentId !== undefined && nextDocuments.some((item) => item.id === preferredDocumentId)
    ? preferredDocumentId
    : nextDocuments.some((item) => item.id === selectedDocumentId.value) ? selectedDocumentId.value : nextDocuments[0]?.id ?? null;
  verifiedVersionId.value = null;
  verifiedAt.value = null;
}

async function load(): Promise<void> {
  loading.value = true;
  try {
    const response = await request("/api/knowledge-bases");
    knowledgeBases.value = await response.json() as KnowledgeBase[];
    const nextId = knowledgeBases.value.some((item) => item.id === selectedKnowledgeBaseId.value) ? selectedKnowledgeBaseId.value : knowledgeBases.value[0]?.id ?? null;
    selectedKnowledgeBaseId.value = nextId;
    documents.value = [];
    selectedDocumentId.value = null;
    verifiedVersionId.value = null;
    verifiedAt.value = null;
    if (nextId !== null) await loadDocuments(nextId);
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { loading.value = false; }
}

async function selectKnowledgeBase(id: number): Promise<void> {
  selectedKnowledgeBaseId.value = id;
  documents.value = [];
  selectedDocumentId.value = null;
  verifiedVersionId.value = null;
  verifiedAt.value = null;
  try { await loadDocuments(id); }
  catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
}

async function createKnowledgeBase(): Promise<void> {
  if (!knowledgeBaseName.value.trim()) { emit("notice", "输入内容无效，请检查后重试。"); return; }
  creating.value = true;
  try {
    const response = await request("/api/knowledge-bases", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: knowledgeBaseName.value }) });
    const created = await response.json() as KnowledgeBase;
    knowledgeBaseName.value = "";
    await load();
    await selectKnowledgeBase(created.id);
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { creating.value = false; }
}

function chooseFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  selectedFile.value = input.files?.[0] ?? null;
  documentName.value = selectedFile.value?.name.replace(/\.[^.]+$/, "") ?? "";
}

async function upload(): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  const file = selectedFile.value;
  if (knowledgeBaseId === null || file === null) { emit("notice", "请选择一个 Markdown 文件。"); return; }
  uploading.value = true;
  try {
    const body = new FormData();
    body.append("file", file);
    if (documentName.value.trim()) body.append("document_name", documentName.value);
    const response = await request(`/api/knowledge-bases/${knowledgeBaseId}/documents`, { method: "POST", body });
    const created = await response.json() as Document;
    selectedFile.value = null;
    documentName.value = "";
    await loadDocuments(knowledgeBaseId, created.id);
  } catch (error) {
    emit("notice", error instanceof TypeError || error instanceof KnowledgeRequestError && error.code === "KNOWLEDGE_ADMISSION_FAILED" ? "上传结果不明确，请先刷新文档列表确认后再试。" : error instanceof Error ? error.message : "操作失败，请稍后重试。");
  } finally { uploading.value = false; }
}

async function verifySource(): Promise<void> {
  if (selectedDocument.value === null) return;
  verifying.value = true;
  try {
    const response = await request(`/api/document-versions/${selectedDocument.value.source.document_version_id}/verify-source`, { method: "POST" });
    const result = await response.json() as VerifyResult;
    verifiedVersionId.value = result.verified ? result.document_version_id : null;
    verifiedAt.value = result.verified ? result.verified_at : null;
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { verifying.value = false; }
}

defineExpose({ load });
onMounted(() => void load());
</script>

<template>
  <section class="mx-auto grid w-full max-w-5xl gap-6 px-5 py-8 lg:grid-cols-[15rem_1fr] sm:px-8">
    <aside class="space-y-4">
      <form
        class="space-y-2 rounded-lg border border-stone-200 bg-white p-3"
        @submit.prevent="createKnowledgeBase"
      >
        <label class="block text-sm font-medium">新建知识库<input
          v-model="knowledgeBaseName"
          class="mt-1 w-full rounded border border-stone-300 p-2"
          maxlength="255"
        ></label>
        <Button
          type="submit"
          :disabled="creating"
        >
          {{ creating ? "正在创建…" : "创建知识库" }}
        </Button>
      </form>
      <p
        v-if="loading"
        class="text-sm text-slate-500"
      >
        正在加载…
      </p>
      <p
        v-else-if="knowledgeBases.length === 0"
        class="text-sm text-slate-500"
      >
        还没有知识库。
      </p>
      <button
        v-for="knowledgeBase in knowledgeBases"
        :key="knowledgeBase.id"
        class="block w-full rounded px-3 py-2 text-left text-sm"
        :class="knowledgeBase.id === selectedKnowledgeBaseId ? 'bg-slate-900 text-white' : 'bg-stone-100'"
        @click="selectKnowledgeBase(knowledgeBase.id)"
      >
        {{ knowledgeBase.name }}
      </button>
    </aside>
    <main
      v-if="selectedKnowledgeBaseId !== null"
      class="min-w-0 space-y-5"
    >
      <form
        class="space-y-3 rounded-lg border border-stone-200 bg-white p-4"
        @submit.prevent="upload"
      >
        <h1 class="font-semibold text-slate-900">
          知识文档
        </h1>
        <label class="block text-sm">Markdown 文件<input
          class="mt-1 block"
          type="file"
          accept=".md,text/markdown"
          :disabled="uploading"
          @change="chooseFile"
        ></label>
        <label class="block text-sm">文档名称（可编辑）<input
          v-model="documentName"
          class="mt-1 w-full rounded border border-stone-300 p-2"
          maxlength="255"
          :disabled="uploading"
        ></label>
        <Button
          type="submit"
          :disabled="uploading || selectedFile === null"
        >
          {{ uploading ? "正在上传…" : "上传 Markdown" }}
        </Button>
      </form>
      <p
        v-if="documents.length === 0 && !loading"
        class="rounded-lg border border-dashed border-stone-300 p-6 text-center text-sm text-slate-500"
      >
        这个知识库还没有文档。
      </p>
      <div
        v-else
        class="grid gap-4 md:grid-cols-2"
      >
        <div class="space-y-2">
          <button
            v-for="document in documents"
            :key="document.id"
            class="block w-full rounded border p-3 text-left"
            :class="document.id === selectedDocumentId ? 'border-slate-700 bg-stone-100' : 'border-stone-200 bg-white'"
            @click="selectedDocumentId = document.id; verifiedVersionId = null; verifiedAt = null"
          >
            <span class="block font-medium">{{ document.name }}</span><span class="mt-1 block text-xs text-slate-500">{{ document.source.filename }}</span>
          </button>
        </div>
        <article
          v-if="selectedDocument"
          class="rounded-lg border border-stone-200 bg-white p-4 text-sm"
        >
          <h2 class="font-semibold">
            {{ selectedDocument.name }}
          </h2>
          <dl class="mt-3 space-y-1 text-slate-600">
            <div>
              <dt class="inline">
                上传时间：
              </dt><dd class="inline">
                {{ selectedDocument.source.created_at }}
              </dd>
            </div><div>
              <dt class="inline">
                文件：
              </dt><dd class="inline">
                {{ selectedDocument.source.filename }}
              </dd>
            </div><div>
              <dt class="inline">
                类型：
              </dt><dd class="inline">
                {{ selectedDocument.source.media_type }}
              </dd>
            </div><div>
              <dt class="inline">
                大小：
              </dt><dd class="inline">
                {{ selectedDocument.source.size_bytes }} bytes
              </dd>
            </div><div>
              <dt class="inline">
                SHA-256：
              </dt><dd class="break-all">
                {{ selectedDocument.source.sha256 }}
              </dd>
            </div>
          </dl>
          <Button
            class="mt-4"
            :disabled="verifying"
            @click="verifySource"
          >
            {{ verifying ? "正在验证…" : "验证原始文件" }}
          </Button>
          <p
            class="mt-3 text-sm"
            :class="verifiedVersionId === selectedDocument.source.document_version_id ? 'text-emerald-700' : 'text-slate-500'"
          >
            原文件完整性：{{ verifiedVersionId === selectedDocument.source.document_version_id ? "刚刚验证通过" : "尚未验证" }}
          </p>
          <p
            v-if="verifiedAt && verifiedVersionId === selectedDocument.source.document_version_id"
            class="text-xs text-emerald-700"
          >
            {{ verifiedAt }}
          </p>
        </article>
      </div>
    </main>
    <main
      v-else
      class="rounded-lg border border-dashed border-stone-300 p-8 text-center text-sm text-slate-500"
    >
      创建或选择一个知识库后即可上传 Markdown 文档。
    </main>
  </section>
</template>
