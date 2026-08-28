<script setup lang="ts">
import type { DialogRootEmits, DialogRootProps } from "reka-ui";
import { DialogRoot, useForwardPropsEmits } from "reka-ui";

defineOptions({ name: "UiSheet" });

const props = defineProps<DialogRootProps>();
const emit = defineEmits<DialogRootEmits>();

const forwarded = useForwardPropsEmits(props, emit);
</script>

<template>
  <DialogRoot
    v-slot="slotProps"
    data-slot="sheet"
    v-bind="forwarded"
  >
    <slot v-bind="slotProps" />
  </DialogRoot>
</template>
