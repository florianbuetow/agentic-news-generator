<template>
  <NuxtLink
    v-if="link"
    :to="link"
    class="sidebar-article-link"
  >
    <article class="sidebar-article">
      <h4 class="sidebar-headline">{{ headline }}</h4>
      <p v-if="byline" class="sidebar-byline">By {{ byline }}</p>
      <p class="sidebar-text">{{ text }}</p>
    </article>
  </NuxtLink>
  <article v-else class="sidebar-article">
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
  },
  link: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.sidebar-article-link {
  text-decoration: none;
  color: inherit;
  display: block;
  transition: background-color 0.2s;
}

.sidebar-article-link:hover .sidebar-article {
  background-color: #f5f5f5;
  cursor: pointer;
}
</style>
