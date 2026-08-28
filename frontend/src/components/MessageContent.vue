<script setup lang="ts">
import MarkdownIt from "markdown-it";
import { computed } from "vue";

const props = defineProps<{ content: string }>();

const SAFE_BASE_URL = "https://langley.invalid";

function safeUrl(value: string): URL | null {
  try {
    const parsed = new URL(value, SAFE_BASE_URL);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed : null;
  } catch {
    return null;
  }
}

const markdown = new MarkdownIt({
  html: false,
  linkify: false,
  typographer: false,
});

markdown.disable("image");
markdown.validateLink = (url) => safeUrl(url) !== null;

const defaultLinkOpen =
  markdown.renderer.rules.link_open ??
  ((tokens, index, options, _environment, renderer) =>
    renderer.renderToken(tokens, index, options));

markdown.renderer.rules.link_open = (tokens, index, options, environment, renderer) => {
  const href = String(tokens[index]?.attrGet("href") ?? "");
  const parsed = safeUrl(href);
  if (parsed !== null && parsed.origin !== new URL(SAFE_BASE_URL).origin) {
    tokens[index]?.attrSet("target", "_blank");
    tokens[index]?.attrSet("rel", "noopener noreferrer");
  }
  return defaultLinkOpen(tokens, index, options, environment, renderer);
};

markdown.renderer.rules.table_open = () => '<div class="message-table-scroll"><table>';
markdown.renderer.rules.table_close = () => "</table></div>";

const renderedContent = computed(() => markdown.render(props.content));
</script>

<template>
  <!-- markdown-it owns this HTML boundary with raw HTML disabled and an explicit link policy. -->
  <!-- eslint-disable vue/no-v-html -->
  <div
    class="message-content min-w-0 break-words"
    v-html="renderedContent"
  />
  <!-- eslint-enable vue/no-v-html -->
</template>

<style scoped>
.message-content {
  color: var(--body);
  overflow-wrap: anywhere;
}

.message-content :deep(> :first-child) {
  margin-top: 0;
}

.message-content :deep(> :last-child) {
  margin-bottom: 0;
}

.message-content :deep(p),
.message-content :deep(ul),
.message-content :deep(ol),
.message-content :deep(pre),
.message-content :deep(.message-table-scroll),
.message-content :deep(blockquote) {
  margin: 0.8rem 0;
}

.message-content :deep(h1),
.message-content :deep(h2),
.message-content :deep(h3),
.message-content :deep(h4),
.message-content :deep(h5),
.message-content :deep(h6) {
  margin: 1.4rem 0 0.55rem;
  color: var(--foreground);
  font-weight: 650;
  letter-spacing: -0.015em;
  line-height: 1.35;
}

.message-content :deep(h1) {
  font-size: 1.35rem;
}

.message-content :deep(h2) {
  font-size: 1.18rem;
}

.message-content :deep(h3),
.message-content :deep(h4),
.message-content :deep(h5),
.message-content :deep(h6) {
  font-size: 1rem;
}

.message-content :deep(ul),
.message-content :deep(ol) {
  padding-left: 1.45rem;
}

.message-content :deep(ul) {
  list-style: disc;
}

.message-content :deep(ol) {
  list-style: decimal;
}

.message-content :deep(li + li) {
  margin-top: 0.25rem;
}

.message-content :deep(a) {
  color: var(--primary);
  text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--primary) 40%, transparent);
  text-underline-offset: 0.18em;
}

.message-content :deep(a:hover) {
  color: var(--primary-deep);
}

.message-content :deep(code) {
  border-radius: var(--radius-sm);
  background: var(--accent);
  color: var(--accent-foreground);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.88em;
  padding: 0.12em 0.34em;
}

.message-content :deep(pre) {
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-md);
  background: #1b262a;
  color: #edf2f0;
  padding: 0.9rem 1rem;
  line-height: 1.6;
}

.message-content :deep(pre code) {
  display: block;
  min-width: max-content;
  border-radius: 0;
  background: transparent;
  color: inherit;
  padding: 0;
}

.message-content :deep(.message-table-scroll) {
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.message-content :deep(table) {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 0.92em;
}

.message-content :deep(th),
.message-content :deep(td) {
  border-bottom: 1px solid var(--border);
  padding: 0.55rem 0.7rem;
  text-align: left;
  vertical-align: top;
}

.message-content :deep(th) {
  background: var(--subtle);
  color: var(--foreground);
  font-weight: 600;
}

.message-content :deep(tr:last-child td) {
  border-bottom: 0;
}

.message-content :deep(blockquote) {
  border-left: 2px solid var(--primary);
  color: var(--muted-foreground);
  padding-left: 0.9rem;
}
</style>
