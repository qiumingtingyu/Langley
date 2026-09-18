<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import { Button } from "@/components/ui/button";
import type { PdfPageRegion, SourceRegion } from "@/types";

type KnowledgeBase = { id: number; name: string; created_at: string };
type DocumentSource = { document_version_id: number; filename: string; media_type: string; size_bytes: number; sha256: string; created_at: string };
type Document = { id: number; name: string; created_at: string; source: DocumentSource };
type VerifyResult = { document_version_id: number; verified: boolean; verified_at: string };
type IndexJob = { id: number; status: string; stage: string | null; processed_chunk_count: number; total_chunk_count: number; error_code: string | null; error_message: string | null; created_at: string; started_at: string | null; finished_at: string | null };
type IndexStatus = { index_status: string; latest_job: IndexJob | null };
type Chunk = { ordinal: number; content: string; heading_path: string[]; source_regions: SourceRegion[] };
type ChunksResponse = { document_version_id: number; successful_chunk_max_chars: number | null; suggested_chunk_max_chars: number; chunk_count: number; offset: number; limit: number; chunks: Chunk[] };
type RebuildResponse = { document_version_id: number; successful_chunk_max_chars: number; chunk_count: number; resulting_index_status: string };
type DocumentProcessingAttempt = { id: number; attempt_no: number; status: string; stage: string | null; recipe_id: string; error_code: string | null; error_message: string | null; created_at: string; started_at: string | null; finished_at: string | null };
type DocumentProcessingStatus = { document_version_id: number; latest_attempt: DocumentProcessingAttempt | null; published_chunks_exist: boolean };

const CHUNK_PAGE_SIZE = 10;
const INDEX_POLL_INTERVAL_MS = 5000;
const PROCESSING_POLL_INTERVAL_MS = 5000;

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
const uploadExpanded = ref<boolean | null>(null);
const uploadError = ref("");
const uploadNotice = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const uploadVisible = computed(() => (uploadExpanded.value ?? documents.value.length === 0) || uploading.value || uploadError.value !== "");
const verifying = ref(false);
const verifiedVersionId = ref<number | null>(null);
const verifiedAt = ref<string | null>(null);
const indexStatus = ref<IndexStatus | null>(null);
const startingIndexBuild = ref(false);
const successfulChunkMaxChars = ref<number | null>(null);
const suggestedChunkMaxChars = ref<number | null>(null);
const draftMaxChunkChars = ref("");
const chunkCount = ref(0);
const chunkOffset = ref(0);
const chunks = ref<Chunk[]>([]);
const chunksLoading = ref(false);
const processingPending = ref(false);
const processingError = ref("");
const chunkLoadError = ref("");
const expandedChunkOrdinals = ref<number[]>([]);
const documentProcessingStatus = ref<DocumentProcessingStatus | null>(null);
const documentProcessingLoading = ref(false);
const documentProcessingError = ref("");
let indexPollTimer: number | null = null;
let processingPollTimer: number | null = null;
let documentSelectionRevision = 0;

const selectedDocument = computed(() => documents.value.find((item) => item.id === selectedDocumentId.value) ?? null);
const selectedDocumentIsPdf = computed(() => selectedDocument.value?.source.media_type === "application/pdf");
const chunksRange = computed(() => chunkCount.value === 0 ? "显示 0–0" : `显示 ${chunkOffset.value + 1}–${Math.min(chunkOffset.value + chunks.value.length, chunkCount.value)}`);
const processingBlockedByIndex = computed(() => indexStatus.value?.index_status === "INDEXING");

function currentIndexStatusText(status: string | undefined): string {
  if (status === "READY") return "可检索";
  if (status === "STALE") return "需要重建";
  if (status === "CHUNKED") return "正在准备检索";
  if (status === "INDEXING") return "正在重建全部索引";
  if (status === "FAILED") return "索引准备失败";
  return status ?? "正在读取";
}

function errorCode(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return;
  const detail = payload.detail;
  if (typeof detail !== "object" || detail === null || !("code" in detail)) return;
  return typeof detail.code === "string" ? detail.code : undefined;
}

function errorMessage(payload: unknown): string {
  const code = errorCode(payload);
  if (code === "KNOWLEDGE_BASE_NOT_FOUND" || code === "DOCUMENT_VERSION_NOT_FOUND") return "该知识库或文档已不存在，请刷新后重试。";
  if (code === "UPLOAD_TOO_LARGE") return "文件超过允许大小。Markdown 最大 5 MiB，PDF 最大 64 MiB。";
  if (code === "UNSUPPORTED_MEDIA_TYPE") return "当前仅支持 Markdown 和 PDF 文件。";
  if (code === "EMPTY_SOURCE") return "文件内容不能为空。";
  if (code === "INVALID_SOURCE_ENCODING") return "文件必须是 UTF-8 编码的 Markdown。";
  if (code === "SOURCE_MISSING") return "原始文件缺失，无法验证。";
  if (code === "SOURCE_INTEGRITY_MISMATCH") return "原始文件与保存时的完整性信息不一致。";
  if (code === "SOURCE_STORAGE_FAILED") return "原始文件存储暂时不可用，请稍后重试。";
  if (code === "KNOWLEDGE_BASE_NOT_CHUNKED") return "当前知识库还没有可索引的分块。";
  if (code === "INDEX_BUILD_IN_PROGRESS") return "该知识库正在建立索引。";
  if (code === "KNOWLEDGE_BASE_INDEXING") return "知识库正在建立索引，请等待完成后重试。";
  if (code === "KNOWLEDGE_BASE_DOCUMENTS_INDEXING") return "部分资料正在准备检索，请稍后再重建全部索引。";
  if (code === "KNOWLEDGE_BASE_DOCUMENTS_PROCESSING") return "还有 PDF 正在处理，请完成后再重建全部索引。";
  if (code === "KNOWLEDGE_BASE_DOCUMENTS_UNPROCESSED") return "还有文档尚未处理，请先完成文档处理后再建立索引。";
  if (code === "VALIDATION_ERROR") return "输入内容无效，请检查后重试。";
  return "操作失败，请稍后重试。";
}

class KnowledgeRequestError extends Error { constructor(readonly code: string | undefined, message: string) { super(message); } }

async function request(path: string, init?: Parameters<typeof window.fetch>[1]): Promise<Response> {
  let response: Response;
  try { response = await fetch(path, init); }
  catch { throw new KnowledgeRequestError("NETWORK_ERROR", "网络连接中断，请检查连接后刷新。"); }
  if (!response.ok) {
    let payload: unknown;
    try { payload = await response.json(); }
    catch { throw new KnowledgeRequestError(undefined, `服务请求未完成（HTTP ${response.status}），请稍后重试。`); }
    throw new KnowledgeRequestError(errorCode(payload), errorMessage(payload));
  }
  return response;
}

async function loadDocuments(knowledgeBaseId: number, preferredDocumentId?: number): Promise<void> {
  const response = await request(`/api/knowledge-bases/${knowledgeBaseId}/documents`);
  const nextDocuments = await response.json() as Document[];
  if (selectedKnowledgeBaseId.value !== knowledgeBaseId) return;
  documents.value = nextDocuments;
  const nextDocumentId = preferredDocumentId !== undefined && nextDocuments.some((item) => item.id === preferredDocumentId)
    ? preferredDocumentId
    : nextDocuments.some((item) => item.id === selectedDocumentId.value) ? selectedDocumentId.value : nextDocuments[0]?.id ?? null;
  await selectDocument(nextDocumentId);
  verifiedVersionId.value = null;
  verifiedAt.value = null;
}

function resetChunkState(): void {
  successfulChunkMaxChars.value = null;
  suggestedChunkMaxChars.value = null;
  draftMaxChunkChars.value = "";
  chunkCount.value = 0;
  chunkOffset.value = 0;
  chunks.value = [];
  chunksLoading.value = false;
  processingError.value = "";
  chunkLoadError.value = "";
  expandedChunkOrdinals.value = [];
}

function isCurrentDocument(versionId: number, revision: number): boolean {
  return revision === documentSelectionRevision && selectedDocument.value?.source.document_version_id === versionId;
}

type ChunkLoadMode = "selection" | "page" | "after-rebuild";

async function loadChunks(offset = 0, mode: ChunkLoadMode = "page"): Promise<void> {
  const versionId = selectedDocument.value?.source.document_version_id;
  if (versionId === undefined) return;
  const revision = documentSelectionRevision;
  chunksLoading.value = true;
  try {
    const response = await request(`/api/document-versions/${versionId}/chunks?offset=${offset}&limit=${CHUNK_PAGE_SIZE}`);
    const page = await response.json() as ChunksResponse;
    if (!isCurrentDocument(versionId, revision)) return;
    successfulChunkMaxChars.value = page.successful_chunk_max_chars;
    suggestedChunkMaxChars.value = page.suggested_chunk_max_chars;
    if (mode === "selection") {
      draftMaxChunkChars.value = String(page.successful_chunk_max_chars ?? page.suggested_chunk_max_chars);
    }
    chunkCount.value = page.chunk_count;
    chunkOffset.value = page.offset;
    chunks.value = page.chunks ?? [];
    expandedChunkOrdinals.value = [];
    chunkLoadError.value = "";
  } catch (error) {
    if (isCurrentDocument(versionId, revision)) {
      const message = error instanceof Error ? error.message : "文档分块暂时不可用，请刷新后重试。";
      if (mode === "selection") {
        resetChunkState();
        if (error instanceof KnowledgeRequestError && error.code === "DOCUMENT_VERSION_NOT_FOUND") {
          const knowledgeBaseId = selectedKnowledgeBaseId.value;
          if (knowledgeBaseId !== null) await loadDocuments(knowledgeBaseId);
        }
        emit("notice", message);
      } else if (mode === "after-rebuild") {
        chunks.value = [];
        chunkLoadError.value = "处理已完成，但暂时无法刷新 Chunk，请刷新后查看。";
      } else {
        chunkLoadError.value = "暂时无法刷新 Chunk，请重试。";
      }
    }
  } finally {
    if (isCurrentDocument(versionId, revision)) chunksLoading.value = false;
  }
}

async function selectDocument(documentId: number | null): Promise<void> {
  documentSelectionRevision += 1;
  selectedDocumentId.value = documentId;
  processingPending.value = false;
  stopDocumentProcessingPolling();
  documentProcessingStatus.value = null;
  documentProcessingLoading.value = false;
  documentProcessingError.value = "";
  resetChunkState();
  if (documentId !== null) {
    await loadChunks(0, "selection");
    if (selectedDocumentIsPdf.value) await loadDocumentProcessingStatus();
  }
}

function validDraftMaxChunkChars(): number | null {
  const value = Number(draftMaxChunkChars.value);
  return Number.isInteger(value) && value > 0 ? value : null;
}

async function rebuildChunks(): Promise<void> {
  if (selectedDocumentIsPdf.value || processingPending.value) return;
  const versionId = selectedDocument.value?.source.document_version_id;
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  const maxChunkChars = validDraftMaxChunkChars();
  if (versionId === undefined || maxChunkChars === null || processingBlockedByIndex.value) {
    if (maxChunkChars === null) processingError.value = "max_chunk_chars 必须是正整数。";
    return;
  }
  const revision = documentSelectionRevision;
  processingPending.value = true;
  processingError.value = "";
  try {
    const response = await request(`/api/document-versions/${versionId}/chunks/rebuild`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_chunk_chars: maxChunkChars }),
    });
    const result = await response.json() as RebuildResponse;
    if (isCurrentDocument(versionId, revision)) {
      successfulChunkMaxChars.value = result.successful_chunk_max_chars;
      draftMaxChunkChars.value = String(result.successful_chunk_max_chars);
      chunkCount.value = result.chunk_count;
      chunkOffset.value = 0;
      chunks.value = [];
      indexStatus.value = indexStatus.value === null ? null : { ...indexStatus.value, index_status: result.resulting_index_status };
      await loadChunks(0, "after-rebuild");
    }
    if (knowledgeBaseId !== null && selectedKnowledgeBaseId.value === knowledgeBaseId) await loadIndexStatus(knowledgeBaseId, true);
  } catch (error) {
    if (!isCurrentDocument(versionId, revision)) {
      if (knowledgeBaseId !== null && selectedKnowledgeBaseId.value === knowledgeBaseId) await loadIndexStatus(knowledgeBaseId);
      return;
    }
    if (error instanceof KnowledgeRequestError && error.code === "KNOWLEDGE_BASE_INDEXING") {
      processingError.value = "知识库正在建立索引，本次重新切片未生效。";
      if (knowledgeBaseId !== null && selectedKnowledgeBaseId.value === knowledgeBaseId) await loadIndexStatus(knowledgeBaseId);
    } else processingError.value = error instanceof Error ? error.message : "文档处理失败，请稍后重试。";
  } finally {
    if (isCurrentDocument(versionId, revision)) processingPending.value = false;
  }
}

function sourceRegionText(region: SourceRegion): string {
  if (region.kind === "text_span") return `text_span [${region.start_byte}, ${region.end_byte})`;
  return `pdf_page [${region.page_start}, ${region.page_end}]`;
}

function sourceFormat(mediaType: string): string {
  if (mediaType === "text/markdown") return "Markdown";
  if (mediaType === "application/pdf") return "PDF";
  return "未知格式";
}

function readableSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "大小未知";
  if (bytes < 1024) return `${bytes} B`;
  const unit = bytes < 1024 * 1024 ? "KiB" : "MiB";
  return `${Number((bytes / (unit === "KiB" ? 1024 : 1024 * 1024)).toFixed(2))} ${unit}`;
}

function localDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "时间未知" : date.toLocaleString("zh-CN", { year: "numeric", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

function fileSummary(source: DocumentSource): string {
  return `${sourceFormat(source.media_type)} · ${readableSize(source.size_bytes)} · ${localDate(source.created_at)}`;
}

function validateFile(file: File): string {
  // Match the API's suffix-based limit; MIME alone is not authoritative.
  const pdf = file.name.toLowerCase().endsWith(".pdf");
  const limit = (pdf ? 64 : 5) * 1024 * 1024;
  if (file.size > limit) return `文件大小 ${readableSize(file.size)}（${file.size.toLocaleString("zh-CN")} bytes），超过 ${pdf ? "PDF 的 64" : "Markdown 的 5"} MiB 限制。`;
  if (pdf && file.type !== "" && file.type !== "application/pdf") return "PDF 文件类型不匹配，请重新选择 PDF 文件。";
  return "";
}

function isPdfPageRegion(region: SourceRegion): region is PdfPageRegion {
  return region.kind === "pdf_page" && Number.isInteger(region.page_start) && Number.isInteger(region.page_end);
}

function pdfPageLocation(regions: SourceRegion[]): string | null {
  const pageRegions = regions.filter(isPdfPageRegion);
  if (pageRegions.length === 0) return null;
  return pageRegions.map((region) => region.page_start === region.page_end ? `第 ${region.page_start} 页` : `第 ${region.page_start}–${region.page_end} 页`).join("、");
}

function processingStatusText(status: DocumentProcessingStatus | null): string {
  const attempt = status?.latest_attempt;
  if (attempt === null || attempt === undefined) return "正在等待处理状态";
  if (attempt.status === "PENDING") return "等待处理";
  if (attempt.status === "RUNNING") {
    if (attempt.stage === "VERIFYING_SOURCE") return "正在校验文件";
    if (attempt.stage === "PARSING") return "正在解析 PDF";
    if (attempt.stage === "CHUNKING") return "正在生成分块";
    if (attempt.stage === "VALIDATING") return "正在校验处理结果";
    if (attempt.stage === "PUBLISHING") return "正在发布资料";
    return "正在处理 PDF";
  }
  if (attempt.status === "SUCCEEDED") return "PDF 已完成处理";
  if (attempt.status === "FAILED") return "PDF 处理失败";
  if (attempt.status === "INTERRUPTED") return "PDF 处理已中断";
  return "正在读取处理状态";
}

function toggleChunk(ordinal: number): void {
  expandedChunkOrdinals.value = expandedChunkOrdinals.value.includes(ordinal)
    ? expandedChunkOrdinals.value.filter((value) => value !== ordinal)
    : [...expandedChunkOrdinals.value, ordinal];
}

function previousChunkPage(): void {
  void loadChunks(Math.max(0, chunkOffset.value - CHUNK_PAGE_SIZE));
}

function nextChunkPage(): void {
  void loadChunks(chunkOffset.value + CHUNK_PAGE_SIZE);
}

function clearIndexPollTimer(): void {
  if (indexPollTimer !== null) window.clearTimeout(indexPollTimer);
  indexPollTimer = null;
}

let preparingReadinessWatchKnowledgeBaseId: number | null = null;

function stopIndexPolling(): void {
  clearIndexPollTimer();
  preparingReadinessWatchKnowledgeBaseId = null;
}

function stopDocumentProcessingPolling(): void {
  if (processingPollTimer !== null) window.clearTimeout(processingPollTimer);
  processingPollTimer = null;
}

async function loadDocumentProcessingStatus(): Promise<void> {
  const versionId = selectedDocument.value?.source.document_version_id;
  if (versionId === undefined || !selectedDocumentIsPdf.value) return;
  const revision = documentSelectionRevision;
  documentProcessingLoading.value = true;
  try {
    const response = await request(`/api/document-versions/${versionId}/processing-status`);
    const next = await response.json() as DocumentProcessingStatus;
    if (!isCurrentDocument(versionId, revision)) return;
    const previousStatus = documentProcessingStatus.value?.latest_attempt?.status;
    documentProcessingStatus.value = next;
    documentProcessingError.value = "";
    stopDocumentProcessingPolling();
    if (next.latest_attempt?.status === "PENDING" || next.latest_attempt?.status === "RUNNING") {
      processingPollTimer = window.setTimeout(() => void loadDocumentProcessingStatus(), PROCESSING_POLL_INTERVAL_MS);
    } else if (previousStatus !== "SUCCEEDED" && next.latest_attempt?.status === "SUCCEEDED") {
      await loadChunks(0, "selection");
      if (!isCurrentDocument(versionId, revision)) return;
      const knowledgeBaseId = selectedKnowledgeBaseId.value;
      if (knowledgeBaseId !== null) await loadIndexStatus(knowledgeBaseId, true);
    }
  } catch (error) {
    if (isCurrentDocument(versionId, revision)) {
      documentProcessingError.value = error instanceof Error ? error.message : "暂时无法读取 PDF 处理状态。";
      stopDocumentProcessingPolling();
    }
  } finally {
    if (isCurrentDocument(versionId, revision)) documentProcessingLoading.value = false;
  }
}

async function loadIndexStatus(knowledgeBaseId: number, watchPreparing = false): Promise<void> {
  if (watchPreparing) preparingReadinessWatchKnowledgeBaseId = knowledgeBaseId;
  const response = await request(`/api/knowledge-bases/${knowledgeBaseId}/index-status`);
  const next = await response.json() as IndexStatus;
  if (selectedKnowledgeBaseId.value !== knowledgeBaseId) return;
  indexStatus.value = next;
  clearIndexPollTimer();
  const hasActiveManualBuild = next.latest_job?.status === "PENDING" || next.latest_job?.status === "RUNNING";
  if (next.index_status !== "CHUNKED" && preparingReadinessWatchKnowledgeBaseId === knowledgeBaseId) {
    preparingReadinessWatchKnowledgeBaseId = null;
  }
  const keepWatchingPreparing = preparingReadinessWatchKnowledgeBaseId === knowledgeBaseId && next.index_status === "CHUNKED";
  if (hasActiveManualBuild || keepWatchingPreparing) {
    indexPollTimer = window.setTimeout(() => void loadIndexStatus(knowledgeBaseId, keepWatchingPreparing).catch((error: unknown) => emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。")), INDEX_POLL_INTERVAL_MS);
  }
}

async function load(): Promise<void> {
  loading.value = true;
  stopIndexPolling();
  try {
    const response = await request("/api/knowledge-bases");
    knowledgeBases.value = await response.json() as KnowledgeBase[];
    const nextId = knowledgeBases.value.some((item) => item.id === selectedKnowledgeBaseId.value) ? selectedKnowledgeBaseId.value : knowledgeBases.value[0]?.id ?? null;
    selectedKnowledgeBaseId.value = nextId;
    documents.value = [];
    selectedDocumentId.value = null;
    indexStatus.value = null;
    stopDocumentProcessingPolling();
    documentProcessingStatus.value = null;
    documentProcessingError.value = "";
    verifiedVersionId.value = null;
    verifiedAt.value = null;
    if (nextId !== null) { await loadDocuments(nextId); await loadIndexStatus(nextId); }
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { loading.value = false; }
}

async function selectKnowledgeBase(id: number): Promise<void> {
  selectedKnowledgeBaseId.value = id;
  documents.value = [];
  selectedDocumentId.value = null;
  indexStatus.value = null;
  stopIndexPolling();
  stopDocumentProcessingPolling();
  documentProcessingStatus.value = null;
  documentProcessingError.value = "";
  verifiedVersionId.value = null;
  verifiedAt.value = null;
  try { await loadDocuments(id); await loadIndexStatus(id); }
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
  if (uploading.value) return;
  const input = event.target as HTMLInputElement;
  selectedFile.value = input.files?.[0] ?? null;
  documentName.value = selectedFile.value?.name.replace(/\.[^.]+$/, "") ?? "";
  uploadError.value = selectedFile.value ? validateFile(selectedFile.value) : "";
  uploadNotice.value = "";
}

async function upload(): Promise<void> {
  if (uploading.value) return;
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  const file = selectedFile.value;
  if (knowledgeBaseId === null || file === null) { uploadError.value = "请选择一个 Markdown 或 PDF 文件。"; return; }
  uploadError.value = validateFile(file);
  if (uploadError.value) return;
  uploadNotice.value = "";
  uploading.value = true;
  let accepted = false;
  try {
    const body = new FormData();
    body.append("file", file.name.toLowerCase().endsWith(".pdf") && !file.type ? new File([file], file.name, { type: "application/pdf" }) : file);
    if (documentName.value.trim()) body.append("document_name", documentName.value);
    const response = await request(`/api/knowledge-bases/${knowledgeBaseId}/documents`, { method: "POST", body });
    accepted = true;
    const created = await response.json() as Document;
    selectedFile.value = null;
    documentName.value = "";
    if (fileInput.value) fileInput.value.value = "";
    uploadNotice.value = response.status === 202 ? "上传请求已接受，PDF 将在后台处理；可检索状态以知识库状态为准。" : "上传请求已接受，请查看文档处理与知识库检索状态。";
    await loadDocuments(knowledgeBaseId, created.id);
    await loadIndexStatus(knowledgeBaseId);
  } catch (error) {
    if (accepted) {
      uploadError.value = "上传请求已接受，但暂时无法刷新资料，请先刷新文档列表确认。";
    } else if (error instanceof KnowledgeRequestError && error.code !== "NETWORK_ERROR" && error.code !== "KNOWLEDGE_ADMISSION_FAILED") {
      uploadError.value = error.message;
    } else {
      uploadError.value = "上传结果不明确，请先刷新文档列表确认后再试。";
    }
    uploadExpanded.value = true;
    emit("notice", uploadError.value);
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

async function startIndexBuild(): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (knowledgeBaseId === null) return;
  startingIndexBuild.value = true;
  try {
    await request(`/api/knowledge-bases/${knowledgeBaseId}/index-build`, { method: "POST" });
    await loadIndexStatus(knowledgeBaseId);
  } catch (error) { emit("notice", error instanceof Error ? error.message : "操作失败，请稍后重试。"); }
  finally { startingIndexBuild.value = false; }
}

defineExpose({ load });
onMounted(() => void load());
onBeforeUnmount(() => {
  stopIndexPolling();
  stopDocumentProcessingPolling();
});
</script>

<template>
  <section class="knowledge-workspace min-h-0 w-full bg-workspace">
    <aside class="knowledge-rail flex min-h-0 flex-col border-b border-border bg-sidebar px-4 py-5 lg:border-b-0 lg:border-r">
      <div class="mb-5 px-1">
        <h1 class="text-xl font-semibold text-foreground">
          知识库
        </h1>
      </div>
      <form
        class="order-3 mt-5 space-y-2 border-t border-border pt-4"
        @submit.prevent="createKnowledgeBase"
      >
        <label class="block text-xs font-medium text-body">新建知识库<input
          v-model="knowledgeBaseName"
          class="mt-1 w-full rounded-sm border border-strong-border bg-surface px-2 py-1.5 text-sm text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
          maxlength="255"
        ></label>
        <Button
          type="submit"
          :disabled="creating"
        >
          {{ creating ? "正在创建…" : "新建知识库" }}
        </Button>
      </form>
      <p
        v-if="loading"
        class="order-1 text-sm text-muted-foreground"
      >
        正在加载…
      </p>
      <p
        v-else-if="knowledgeBases.length === 0"
        class="order-1 text-sm text-muted-foreground"
      >
        还没有知识库。
      </p>
      <button
        v-for="knowledgeBase in knowledgeBases"
        :key="knowledgeBase.id"
        class="order-2 block w-full shrink-0 break-words border-l-2 px-3 py-2 text-left text-sm transition-colors focus-visible:outline focus-visible:outline-ring"
        :class="knowledgeBase.id === selectedKnowledgeBaseId ? 'border-primary bg-surface font-medium! text-foreground' : 'border-transparent text-muted-foreground hover:bg-subtle hover:text-foreground'"
        @click="selectKnowledgeBase(knowledgeBase.id)"
      >
        {{ knowledgeBase.name }}
      </button>
    </aside>
    <main
      v-if="selectedKnowledgeBaseId !== null"
      class="knowledge-main min-w-0 bg-workspace"
    >
      <header class="shrink-0 border-b border-border bg-workspace px-5 py-4 sm:px-6">
        <div class="flex flex-wrap items-start justify-between gap-4">
          <div class="min-w-0">
            <h2 class="mt-1 break-words text-xl font-semibold tracking-[-0.02em] text-foreground">
              {{ knowledgeBases.find((item) => item.id === selectedKnowledgeBaseId)?.name }}
            </h2>
            <p class="mt-2 text-sm text-muted-foreground">
              索引状态：{{ currentIndexStatusText(indexStatus?.index_status) }}
            </p>
          </div>
          <Button
            type="button"
            :aria-expanded="uploadVisible"
            aria-controls="knowledge-upload"
            :disabled="uploading"
            @click="uploadExpanded = !uploadVisible"
          >
            {{ uploadVisible ? "收起上传" : "添加资料" }}
          </Button>
        </div>
        <details class="mt-2 text-xs text-muted-foreground">
          <summary class="cursor-pointer">
            维护操作
          </summary>
          <div class="mt-3 flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="small"
              :disabled="startingIndexBuild || indexStatus?.index_status === 'INDEXING'"
              @click="startIndexBuild"
            >
              {{ startingIndexBuild ? "正在提交…" : "重建全部索引" }}
            </Button>
            <p>用于全库索引维护；资料是否可检索以当前状态为准。</p>
          </div>
        </details>
        <div
          v-if="indexStatus?.latest_job && (indexStatus.latest_job.status === 'PENDING' || indexStatus.latest_job.status === 'RUNNING' || indexStatus.latest_job.status === 'FAILED' || indexStatus.latest_job.status === 'INTERRUPTED')"
          class="mt-4 border-l-2 border-primary bg-subtle px-3 py-2 text-sm text-body"
        >
          <p v-if="indexStatus.latest_job.status === 'PENDING' || indexStatus.latest_job.status === 'RUNNING'">
            正在重建全部索引 · {{ indexStatus.latest_job.stage ?? "等待执行" }} · {{ indexStatus.latest_job.processed_chunk_count }} / {{ indexStatus.latest_job.total_chunk_count }}
          </p>
          <p v-else>
            {{ indexStatus.latest_job.error_message ?? "索引建立失败，请重试。" }}
          </p>
        </div>
      </header>
      <div class="knowledge-body flex min-h-0 flex-col gap-4 px-5 py-4 sm:px-6">
        <form
          v-show="uploadVisible"
          id="knowledge-upload"
          class="upload-form flex flex-col gap-3 border-b border-border pb-4"
          @submit.prevent="upload"
        >
          <h3 class="text-sm font-semibold text-foreground">
            添加资料
          </h3>
          <p class="text-sm text-muted-foreground">
            支持 Markdown 和 PDF。<br>Markdown 最大 5 MiB · PDF 最大 64 MiB
          </p>
          <p
            v-if="uploadError"
            id="upload-error"
            role="alert"
            class="break-words rounded-md border border-danger-border bg-danger-surface px-3 py-2 text-sm text-danger-foreground"
          >
            {{ uploadError }}
          </p>
          <label class="inline-flex cursor-pointer items-center rounded-sm border border-strong-border bg-surface px-3 py-2 text-sm font-medium text-body hover:bg-subtle focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-ring">选择文件<input
            ref="fileInput"
            class="sr-only"
            type="file"
            :aria-invalid="uploadError !== ''"
            :aria-describedby="uploadError ? 'upload-error' : undefined"
            accept=".md,.pdf,text/markdown,application/pdf"
            :disabled="uploading"
            @change="chooseFile"
          ></label>
          <p class="break-all text-sm text-muted-foreground">
            {{ selectedFile === null ? "未选择文件" : `已选择：${selectedFile.name}` }}
          </p>
          <label class="block text-sm text-body">文档名称（可编辑）<input
            v-model="documentName"
            class="mt-1 w-full rounded-sm border border-strong-border bg-surface px-2 py-1.5 text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
            maxlength="255"
            :disabled="uploading"
          ></label>
          <div class="flex flex-wrap items-center gap-3">
            <Button
              type="submit"
              :disabled="uploading || selectedFile === null || !!(selectedFile && validateFile(selectedFile))"
            >
              {{ uploading ? "正在上传…" : "上传资料" }}
            </Button>
            <Button
              type="button"
              variant="ghost"
              :disabled="uploading || loading"
              @click="load"
            >
              刷新文档列表
            </Button>
          </div>
        </form>
        <p
          v-if="uploadNotice"
          role="status"
          class="text-sm text-body"
        >
          {{ uploadNotice }}
        </p>
        <p
          v-if="documents.length === 0 && !loading"
          class="border border-dashed border-strong-border px-5 py-10 text-center text-sm text-muted-foreground"
        >
          这个知识库还没有文档。
        </p>
        <div
          v-else
          class="document-workspace grid min-h-0 gap-5"
        >
          <section
            class="document-list min-w-0 border-b border-border pb-4"
            aria-label="文档列表"
            tabindex="0"
          >
            <h3 class="mb-3 text-sm font-semibold text-foreground">
              文档
            </h3>
            <button
              v-for="document in documents"
              :key="document.id"
              class="block w-full border-l-2 px-3 py-3 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              :class="document.id === selectedDocumentId ? 'border-primary bg-subtle text-foreground' : 'border-transparent text-body hover:bg-subtle'"
              @click="selectDocument(document.id); verifiedVersionId = null; verifiedAt = null"
            >
              <span class="block break-words font-medium">{{ document.name }}</span><span class="mt-1 block break-all text-xs text-muted-foreground">{{ sourceFormat(document.source.media_type) }} · {{ readableSize(document.source.size_bytes) }}</span>
            </button>
          </section>
          <article
            v-if="selectedDocument"
            class="document-detail min-w-0 text-sm"
            aria-label="文档阅读区"
            tabindex="0"
          >
            <h3 class="mt-1 break-words text-lg font-medium text-foreground">
              {{ selectedDocument.name }}
            </h3>
            <p class="mt-3 text-xs font-normal leading-5 text-muted-foreground">
              {{ fileSummary(selectedDocument.source) }}
            </p>
            <p class="mt-1 break-all text-xs leading-5 text-muted-foreground">
              {{ selectedDocument.source.filename }}
            </p>

            <section
              v-if="!selectedDocumentIsPdf"
              class="mt-6 border-t border-border pt-5"
            >
              <h4
                v-if="successfulChunkMaxChars === null"
                class="font-semibold text-foreground"
              >
                文档处理
              </h4>
              <p
                v-if="successfulChunkMaxChars === null && chunkCount === 0 && !chunksLoading"
                class="mt-2 text-muted-foreground"
              >
                尚未处理
              </p>
              <p
                v-else-if="successfulChunkMaxChars === null && !chunksLoading"
                class="mt-2 text-warning-foreground"
              >
                当前切片配置未知，请重新处理文档。
              </p>
              <p
                v-if="processingBlockedByIndex"
                class="mt-2 text-warning-foreground"
              >
                正在建立索引，请等待完成后再重新处理文档。
              </p>
              <p
                v-if="processingError"
                role="alert"
                class="mt-2 text-danger-foreground"
              >
                {{ processingError }}
              </p>
              <details
                :key="selectedDocument.id"
                :open="successfulChunkMaxChars === null"
                class="mt-3"
              >
                <summary class="cursor-pointer text-sm text-body">
                  处理设置
                </summary>
                <p
                  v-if="successfulChunkMaxChars !== null"
                  class="mt-2 text-xs text-muted-foreground"
                >
                  当前分块配置：{{ successfulChunkMaxChars }}
                </p>
                <form
                  class="mt-3 flex flex-wrap items-end gap-3"
                  @submit.prevent="rebuildChunks"
                >
                  <label class="block text-sm">
                    单个分块最大字符数
                    <span class="ml-1 font-mono text-[10px] text-muted-light">max_chunk_chars</span>
                    <input
                      v-model="draftMaxChunkChars"
                      class="mt-1 block w-40 rounded-sm border border-strong-border bg-surface p-2 text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-ring/20"
                      inputmode="numeric"
                      :disabled="processingPending || processingBlockedByIndex"
                    >
                  </label>
                  <Button
                    type="submit"
                    :disabled="processingPending || processingBlockedByIndex"
                  >
                    {{ processingPending ? "处理中…" : successfulChunkMaxChars === null ? "处理文档" : "重新切片" }}
                  </Button>
                </form>
              </details>
            </section>
            <section
              v-else
              class="mt-6 border-t border-border pt-5"
            >
              <h4 class="font-semibold text-foreground">
                PDF 处理
              </h4>
              <p
                class="mt-2 text-body"
                aria-live="polite"
              >
                {{ documentProcessingLoading && documentProcessingStatus === null ? "正在读取处理状态…" : processingStatusText(documentProcessingStatus) }}
              </p>
              <p
                v-if="documentProcessingStatus?.latest_attempt?.status === 'PENDING' || documentProcessingStatus?.latest_attempt?.status === 'RUNNING'"
                class="mt-2 text-sm leading-6 text-muted-foreground"
              >
                PDF 解析可能需要一些时间，可以离开此页面，处理会在后台继续。
              </p>
              <p
                v-if="documentProcessingStatus?.latest_attempt?.status === 'FAILED' || documentProcessingStatus?.latest_attempt?.status === 'INTERRUPTED'"
                class="mt-2 break-words text-sm leading-6 text-muted-foreground"
              >
                {{ documentProcessingStatus.latest_attempt.error_message ?? "请稍后查看处理状态。" }}
              </p>
              <p
                v-if="documentProcessingError"
                role="alert"
                class="mt-2 text-danger-foreground"
              >
                {{ documentProcessingError }}
              </p>
            </section>
            <section class="mt-6 border-t border-border pt-5">
              <div class="flex items-baseline justify-between gap-3">
                <h4 class="font-semibold text-foreground">
                  分块预览
                </h4>
              </div>
              <p
                v-if="chunksLoading"
                class="mt-3 text-muted-foreground"
              >
                正在读取分块…
              </p>
              <p
                v-if="chunkLoadError"
                role="alert"
                class="mt-3 text-danger-foreground"
              >
                {{ chunkLoadError }}
              </p>
              <p
                v-else-if="chunkCount === 0"
                class="mt-3 text-muted-foreground"
              >
                暂无分块。
              </p>
              <div
                v-else
                class="mt-3 space-y-3"
              >
                <div class="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                  <p>共 {{ chunkCount }} 个分块 · {{ chunksRange }}</p>
                  <div class="flex items-center gap-3">
                    <Button
                      type="button"
                      variant="outline"
                      size="small"
                      :disabled="chunksLoading || chunkOffset === 0"
                      @click="previousChunkPage"
                    >
                      上一页
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="small"
                      :disabled="chunksLoading || chunkOffset + chunks.length >= chunkCount"
                      @click="nextChunkPage"
                    >
                      下一页
                    </Button>
                  </div>
                </div>
                <article
                  v-for="chunk in chunks"
                  :key="chunk.ordinal"
                  class="border-t border-border py-5"
                >
                  <p
                    v-if="chunk.heading_path.length > 0"
                    class="mt-1 text-[13px] font-normal text-muted-foreground"
                  >
                    {{ chunk.heading_path.join(" › ") }}
                  </p>
                  <p
                    v-if="selectedDocumentIsPdf && pdfPageLocation(chunk.source_regions)"
                    class="mt-1 text-[13px] font-normal text-muted-foreground"
                  >
                    页码：{{ pdfPageLocation(chunk.source_regions) }}
                  </p>
                  <p class="mt-2 break-words whitespace-pre-wrap text-[15px] font-normal leading-[1.75] text-body">
                    {{ expandedChunkOrdinals.includes(chunk.ordinal) ? chunk.content : `${chunk.content.slice(0, 240)}${chunk.content.length > 240 ? "…" : ""}` }}
                  </p>
                  <button
                    v-if="chunk.content.length > 240"
                    :aria-expanded="expandedChunkOrdinals.includes(chunk.ordinal)"
                    class="mt-2 text-sm text-primary-deep underline underline-offset-2 focus-visible:outline focus-visible:outline-ring"
                    type="button"
                    @click="toggleChunk(chunk.ordinal)"
                  >
                    {{ expandedChunkOrdinals.includes(chunk.ordinal) ? "收起" : "展开" }}
                  </button>
                  <details class="mt-3 text-xs text-muted-foreground">
                    <summary class="cursor-pointer">
                      位置详情 · 片段 {{ chunk.ordinal }}
                    </summary>
                    <p class="mt-1 break-all font-mono">
                      {{ chunk.source_regions.map(sourceRegionText).join("；") }}
                    </p>
                  </details>
                </article>
              </div>
              <div
                v-if="chunkCount > 0"
                class="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground"
              >
                <p>共 {{ chunkCount }} 个分块 · {{ chunksRange }}</p>
                <div class="flex items-center gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    size="small"
                    :disabled="chunksLoading || chunkOffset === 0"
                    @click="previousChunkPage"
                  >
                    上一页
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="small"
                    :disabled="chunksLoading || chunkOffset + chunks.length >= chunkCount"
                    @click="nextChunkPage"
                  >
                    下一页
                  </Button>
                </div>
              </div>
            </section>
            <details class="mt-5 border-t border-border pt-3 text-body">
              <summary class="cursor-pointer text-sm font-medium text-foreground">
                来源与完整性
              </summary>
              <div class="mt-3 space-y-3">
                <p class="break-all font-mono text-xs text-muted-foreground">
                  SHA-256 · {{ selectedDocument.source.sha256 }}
                </p>
                <Button
                  :disabled="verifying"
                  @click="verifySource"
                >
                  {{ verifying ? "正在验证…" : "验证原始文件" }}
                </Button>
                <p class="text-sm text-muted-foreground">
                  原文件完整性：{{ verifiedVersionId === selectedDocument.source.document_version_id ? "刚刚验证通过" : "尚未验证" }}
                </p>
                <p
                  v-if="verifiedAt && verifiedVersionId === selectedDocument.source.document_version_id"
                  class="font-mono text-xs text-muted-foreground"
                >
                  {{ verifiedAt }}
                </p>
              </div>
            </details>
          </article>
        </div>
      </div>
    </main>
    <main
      v-else
      class="m-5 border border-dashed border-strong-border px-6 py-12 text-center text-sm text-muted-foreground sm:m-8"
    >
      创建或选择一个知识库后即可管理资料。当前支持 Markdown 和 PDF 文件。
    </main>
  </section>
</template>


<style scoped>
/* App supplies a bounded desktop viewport; each content pane owns its scroll. */
@media (min-width: 1024px) {
  .knowledge-workspace { display: grid; height: 100%; overflow: hidden; grid-template-columns: 11rem minmax(0, 1fr); }
  .knowledge-rail { overflow-y: auto; }
  .knowledge-main { display: flex; min-height: 0; flex-direction: column; overflow: hidden; }
  .knowledge-body { flex: 1; overflow: hidden; }
  .upload-form { flex-shrink: 0; max-height: 45vh; overflow-y: auto; }
  .document-workspace { flex: 1; grid-template-rows: minmax(7rem, 0.35fr) minmax(0, 1fr); overflow: hidden; }
  .document-list, .document-detail { min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; }
  .document-detail { padding-right: 0.5rem; }
}
@media (min-width: 1280px) {
  .document-workspace { grid-template-columns: 12rem minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); }
  .document-list { border-bottom: 0; border-right: 1px solid var(--border); padding: 0 0.75rem 0 0; }
}
@media (min-width: 1440px) {
  .knowledge-workspace { grid-template-columns: 12rem minmax(0, 1fr); }
  .document-workspace { grid-template-columns: 14rem minmax(0, 1fr); gap: 1.75rem; }
}
.knowledge-workspace summary:focus-visible { outline: 2px solid var(--ring); outline-offset: 3px; }
</style>
