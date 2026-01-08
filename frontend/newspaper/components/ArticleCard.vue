<template>
  <article :class="['article-card', { large: size === 'large' }]">
    <img 
      v-if="image" 
      class="article-card-image" 
      :src="image" 
      :alt="headline"
    >
    <div>
      <div v-if="section" class="article-card-section">{{ section }}</div>
      <NuxtLink v-if="link" :to="link" class="headline-link">
        <h3 class="article-card-headline">{{ headline }}</h3>
      </NuxtLink>
      <h3 v-else class="article-card-headline">{{ headline }}</h3>
      <p v-if="subhead" class="article-card-subhead">{{ subhead }}</p>
      <p v-if="byline" class="article-card-byline">By {{ byline }}</p>
      <p class="article-card-text">{{ text }}</p>
      <NuxtLink v-if="link" :to="link" class="read-more">Continue Reading →</NuxtLink>
    </div>
  </article>
</template>

<script setup>
defineProps({
  image: {
    type: String,
    default: ''
  },
  section: {
    type: String,
    default: ''
  },
  headline: {
    type: String,
    required: true
  },
  subhead: {
    type: String,
    default: ''
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
  },
  size: {
    type: String,
    default: 'normal',
    validator: (value) => ['normal', 'large'].includes(value)
  }
})
</script>
