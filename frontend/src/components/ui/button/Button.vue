<script setup lang="ts">
import { cva } from "class-variance-authority";
import { computed } from "vue";

import { cn } from "@/lib/utils";

defineOptions({ name: "UiButton" });

type ButtonVariant = "accent" | "ghost" | "outline";
type ButtonSize = "default" | "icon" | "small";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        accent:
          "bg-slate-900 text-white hover:bg-slate-700 focus-visible:ring-slate-500",
        ghost:
          "bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-950 focus-visible:ring-slate-400",
        outline:
          "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 focus-visible:ring-slate-400",
      },
      size: {
        default: "h-9 px-3.5 text-sm",
        icon: "size-9 p-0",
        small: "h-8 px-2.5 text-xs",
      },
    },
    defaultVariants: {
      variant: "accent",
      size: "default",
    },
  },
);

const props = withDefaults(
  defineProps<{
    variant?: ButtonVariant;
    size?: ButtonSize;
    type?: "button" | "reset" | "submit";
    disabled?: boolean;
    class?: string;
  }>(),
  {
    variant: "accent",
    size: "default",
    type: "button",
    class: "",
  },
);

const buttonClass = computed(() => {
  return cn(buttonVariants({ variant: props.variant, size: props.size }), props.class);
});
</script>

<template>
  <button
    data-slot="button"
    :class="buttonClass"
    :disabled="disabled"
    :type="type"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>
