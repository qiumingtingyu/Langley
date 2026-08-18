<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{ content: string }>();

type ContentBlock =
  | { kind: "paragraph"; content: string }
  | { kind: "unordered-list"; items: string[] }
  | { kind: "ordered-list"; items: string[] }
  | { kind: "code"; content: string };

const unorderedList = /^\s*[-*+]\s+(.+)$/;
const orderedList = /^\s*\d+\.\s+(.+)$/;

function messageBlocks(content: string): ContentBlock[] {
  const lines = content.replaceAll("\r\n", "\n").split("\n");
  const blocks: ContentBlock[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.trimStart().startsWith("```")) {
      index += 1;
      const codeLines: string[] = [];
      while (index < lines.length && !(lines[index] ?? "").trimStart().startsWith("```")) {
        codeLines.push(lines[index] ?? "");
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({ kind: "code", content: codeLines.join("\n") });
      continue;
    }

    const unordered = line.match(unorderedList);
    const ordered = line.match(orderedList);
    if (unordered !== null || ordered !== null) {
      const kind = unordered === null ? "ordered-list" : "unordered-list";
      const expression = unordered === null ? orderedList : unorderedList;
      const items: string[] = [];
      while (index < lines.length) {
        const item = (lines[index] ?? "").match(expression);
        if (item === null) {
          break;
        }
        items.push(item[1] ?? "");
        index += 1;
      }
      blocks.push({ kind, items });
      continue;
    }

    const paragraph: string[] = [];
    while (index < lines.length) {
      const paragraphLine = lines[index] ?? "";
      if (
        !paragraphLine.trim() ||
        paragraphLine.trimStart().startsWith("```") ||
        unorderedList.test(paragraphLine) ||
        orderedList.test(paragraphLine)
      ) {
        break;
      }
      paragraph.push(paragraphLine);
      index += 1;
    }
    blocks.push({ kind: "paragraph", content: paragraph.join("\n") });
  }

  return blocks;
}

const blocks = computed(() => messageBlocks(props.content));
</script>

<template>
  <div class="space-y-3 break-words">
    <template
      v-for="(block, index) in blocks"
      :key="index"
    >
      <p
        v-if="block.kind === 'paragraph'"
        class="whitespace-pre-wrap"
      >
        {{ block.content }}
      </p>
      <ul
        v-else-if="block.kind === 'unordered-list'"
        class="list-disc space-y-1 pl-6"
      >
        <li
          v-for="(item, itemIndex) in block.items"
          :key="itemIndex"
        >
          {{ item }}
        </li>
      </ul>
      <ol
        v-else-if="block.kind === 'ordered-list'"
        class="list-decimal space-y-1 pl-6"
      >
        <li
          v-for="(item, itemIndex) in block.items"
          :key="itemIndex"
        >
          {{ item }}
        </li>
      </ol>
      <pre
        v-else
        class="max-w-full overflow-x-auto rounded-lg bg-slate-900 p-4 text-sm leading-6 text-slate-100"
      ><code>{{ block.content }}</code></pre>
    </template>
  </div>
</template>
