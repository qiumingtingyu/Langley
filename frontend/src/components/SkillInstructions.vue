<script setup lang="ts">
import MarkdownIt from "markdown-it";
import { computed } from "vue";

const props = defineProps<{ content: string }>();

function safeExternalUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

const markdown = new MarkdownIt({
  html: false,
  linkify: false,
  typographer: false,
});

markdown.disable("image");
markdown.validateLink = safeExternalUrl;

const defaultLinkOpen =
  markdown.renderer.rules.link_open ??
  ((tokens, index, options, _environment, renderer) =>
    renderer.renderToken(tokens, index, options));

markdown.renderer.rules.link_open = (tokens, index, options, environment, renderer) => {
  tokens[index]?.attrSet("target", "_blank");
  tokens[index]?.attrSet("rel", "noopener noreferrer");
  return defaultLinkOpen(tokens, index, options, environment, renderer);
};

markdown.renderer.rules.table_open = () => '<div class="skill-table-scroll"><table>';
markdown.renderer.rules.table_close = () => "</table></div>";

const renderedContent = computed(() => markdown.render(props.content));
</script>

<template>
  <!-- markdown-it owns this boundary with raw HTML and images disabled. -->
  <!-- eslint-disable vue/no-v-html -->
  <div
    class="skill-instructions min-w-0 break-words"
    v-html="renderedContent"
  />
  <!-- eslint-enable vue/no-v-html -->
</template>

<style scoped>
.skill-instructions {
  color: var(--body);
  font-size: 0.9375rem;
  line-height: 1.72;
  overflow-wrap: anywhere;
}

.skill-instructions :deep(> :first-child) { margin-top: 0; }
.skill-instructions :deep(> :last-child) { margin-bottom: 0; }

.skill-instructions :deep(p),
.skill-instructions :deep(ul),
.skill-instructions :deep(ol),
.skill-instructions :deep(pre),
.skill-instructions :deep(.skill-table-scroll),
.skill-instructions :deep(blockquote) {
  margin: 0.8rem 0;
}

.skill-instructions :deep(h1),
.skill-instructions :deep(h2),
.skill-instructions :deep(h3),
.skill-instructions :deep(h4),
.skill-instructions :deep(h5),
.skill-instructions :deep(h6) {
  margin: 1.45rem 0 0.55rem;
  color: var(--foreground);
  font-weight: 650;
  letter-spacing: -0.015em;
  line-height: 1.35;
}

.skill-instructions :deep(h1) { font-size: 1.3rem; }
.skill-instructions :deep(h2) { font-size: 1.15rem; }
.skill-instructions :deep(h3),
.skill-instructions :deep(h4),
.skill-instructions :deep(h5),
.skill-instructions :deep(h6) { font-size: 1rem; }

.skill-instructions :deep(ul),
.skill-instructions :deep(ol) { padding-left: 1.45rem; }
.skill-instructions :deep(ul) { list-style: disc; }
.skill-instructions :deep(ol) { list-style: decimal; }
.skill-instructions :deep(li + li) { margin-top: 0.25rem; }

.skill-instructions :deep(a) {
  color: var(--primary-deep);
  text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--primary) 42%, transparent);
  text-underline-offset: 0.18em;
}

.skill-instructions :deep(code) {
  border-radius: var(--radius-sm);
  background: var(--accent);
  color: var(--accent-foreground);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.88em;
  padding: 0.12em 0.34em;
}

.skill-instructions :deep(pre) {
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--strong-border);
  border-radius: var(--radius-md);
  background: #1b262a;
  color: #edf2f0;
  padding: 0.9rem 1rem;
  line-height: 1.6;
}

.skill-instructions :deep(pre code) {
  display: block;
  min-width: max-content;
  border-radius: 0;
  background: transparent;
  color: inherit;
  padding: 0;
}

.skill-instructions :deep(blockquote) {
  border-left: 2px solid var(--primary);
  color: var(--muted-foreground);
  padding-left: 0.9rem;
}

.skill-instructions :deep(.skill-table-scroll) {
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.skill-instructions :deep(table) {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 0.92em;
}

.skill-instructions :deep(th),
.skill-instructions :deep(td) {
  border-bottom: 1px solid var(--border);
  padding: 0.55rem 0.7rem;
  text-align: left;
  vertical-align: top;
}

.skill-instructions :deep(th) {
  background: var(--subtle);
  color: var(--foreground);
  font-weight: 600;
}

.skill-instructions :deep(tr:last-child td) { border-bottom: 0; }
</style>
