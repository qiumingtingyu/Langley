<script setup lang="ts">
import { computed } from "vue";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { MessageCitation } from "@/types";

const props = defineProps<{
  citation: MessageCitation | null;
}>();

const emit = defineEmits<{
  close: [];
}>();

const ordinal = computed(() =>
  props.citation === null ? "" : String(props.citation.evidence_handle).padStart(2, "0"),
);

const headingSegments = computed(() =>
  (props.citation?.heading_path ?? []).filter(
    (segment): segment is string => typeof segment === "string" && segment.trim().length > 0,
  ),
);

function handleOpenChange(open: boolean): void {
  if (!open) emit("close");
}
</script>

<template>
  <Sheet
    :open="citation !== null"
    @update:open="handleOpenChange"
  >
    <SheetContent
      side="right"
      class="w-full min-w-0 gap-0 border-strong-border bg-surface p-0 text-foreground shadow-lg sm:max-w-[28rem]"
    >
      <template v-if="citation">
        <SheetHeader class="border-b border-border px-6 py-6 pr-14 text-left sm:px-8 sm:py-7 sm:pr-16">
          <p class="font-mono text-[10px] font-medium tracking-[0.16em] text-primary">
            EVIDENCE {{ ordinal }}
          </p>
          <SheetTitle class="mt-3 break-words text-base font-semibold tracking-[-0.01em] text-foreground [overflow-wrap:anywhere]">
            {{ citation.source_display_name }}
          </SheetTitle>
          <p
            v-if="headingSegments.length > 0"
            class="mt-1 break-words text-xs leading-5 text-muted-foreground [overflow-wrap:anywhere]"
          >
            {{ headingSegments.join(" › ") }}
          </p>
          <SheetDescription class="sr-only">
            回答生成时保存的证据快照
          </SheetDescription>
        </SheetHeader>

        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto px-6 py-6 sm:px-8 sm:py-7">
          <div class="flex items-center gap-3">
            <p class="font-mono text-[9px] font-medium tracking-[0.15em] text-muted-light">
              EVIDENCE
            </p>
            <span
              class="h-px flex-1 bg-border"
              aria-hidden="true"
            />
          </div>
          <div class="mt-5 break-words whitespace-pre-wrap text-[15px] leading-7 text-body [overflow-wrap:anywhere]">
            {{ citation.evidence_text }}
          </div>
          <p class="mt-auto border-t border-border pt-5 text-xs leading-5 text-muted-foreground">
            回答生成时的证据快照
          </p>
        </div>
      </template>
    </SheetContent>
  </Sheet>
</template>
