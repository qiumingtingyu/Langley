<script setup lang="ts">
import { reactiveOmit } from "@vueuse/core";
import { X } from "lucide-vue-next";
import type { DialogContentEmits, DialogContentProps } from "reka-ui";
import {
  DialogClose,
  DialogContent,
  DialogPortal,
  useForwardPropsEmits,
} from "reka-ui";
import type { HTMLAttributes } from "vue";

import { cn } from "@/lib/utils";

import SheetOverlay from "./SheetOverlay.vue";

interface SheetContentProps extends DialogContentProps {
  class?: HTMLAttributes["class"];
  side?: "top" | "right" | "bottom" | "left";
}

defineOptions({
  name: "UiSheetContent",
  inheritAttrs: false,
});

const props = withDefaults(defineProps<SheetContentProps>(), {
  class: "",
  side: "right",
});
const emit = defineEmits<DialogContentEmits>();

const delegatedProps = reactiveOmit(props, "class", "side");
const forwarded = useForwardPropsEmits(delegatedProps, emit);
</script>

<template>
  <DialogPortal>
    <SheetOverlay />
    <DialogContent
      data-slot="sheet-content"
      :class="cn(
        'fixed z-50 flex flex-col gap-4 border-border bg-background shadow-lg outline-none',
        side === 'right' && 'inset-y-0 right-0 h-full w-3/4 border-l sm:max-w-sm',
        side === 'left' && 'inset-y-0 left-0 h-full w-3/4 border-r sm:max-w-sm',
        side === 'top' && 'inset-x-0 top-0 h-auto border-b',
        side === 'bottom' && 'inset-x-0 bottom-0 h-auto border-t',
        props.class,
      )"
      v-bind="{ ...$attrs, ...forwarded }"
    >
      <slot />

      <DialogClose
        data-slot="sheet-close"
        aria-label="关闭面板"
        class="absolute right-4 top-4 inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 [&>svg]:size-4"
      >
        <X aria-hidden="true" />
        <span class="sr-only">关闭面板</span>
      </DialogClose>
    </DialogContent>
  </DialogPortal>
</template>
