<template>
  <article class="sidebar-article">
    <h4 class="sidebar-headline">{{ headline }}</h4>
    <p v-if="byline" class="sidebar-byline">By {{ byline }}</p>
    <p class="sidebar-text">{{ text }}</p>
  </article>
</template>

<script setup>
defineProps({
  headline: {
    type: String,
    required: true
  },
  byline: {
    type: String,
    default: ''
  },
  text: {
    type: String,
    required: true
  }
})
</script>
