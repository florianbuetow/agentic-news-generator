<template>
  <div class="brief-item">
    <div class="brief-headline">{{ headline }}</div>
    <div v-if="text" class="brief-text">{{ text }}</div>
  </div>
</template>

<script setup>
defineProps({
  headline: {
    type: String,
    required: true
  },
  text: {
    type: String,
    default: ''
  }
})
</script>
