<template>
  <NuxtLink
    v-if="link"
    :to="link"
    class="brief-item-link"
  >
    <div class="brief-item">
      <div class="brief-headline">{{ headline }}</div>
      <div v-if="text" class="brief-text">{{ text }}</div>
    </div>
  </NuxtLink>
  <div v-else class="brief-item">
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
  },
  link: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.brief-item-link {
  text-decoration: none;
  color: inherit;
  display: block;
  transition: background-color 0.2s;
}

.brief-item-link:hover .brief-item {
  background-color: #f5f5f5;
  cursor: pointer;
}
</style>
