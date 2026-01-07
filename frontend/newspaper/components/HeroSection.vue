<template>
  <section class="hero-section">
    <div class="hero-image">
      <img :src="image" :alt="headline">
      <div class="hero-image-caption">
        <strong>{{ captionLabel }}</strong> {{ caption }}
      </div>
    </div>
    <div class="hero-content">
      <h2 class="hero-headline">{{ headline }}</h2>
      <p v-if="subhead" class="hero-subhead">{{ subhead }}</p>
      <p v-if="byline" class="hero-byline">By {{ byline }}</p>
      <div class="hero-text">
        <p v-for="(paragraph, index) in paragraphs" :key="index">
          <span v-if="index === 0 && dateline" class="dateline">{{ dateline }} —</span>
          {{ paragraph }}
        </p>
      </div>
      <NuxtLink v-if="link" :to="link" class="read-more">Continue Reading →</NuxtLink>
    </div>
  </section>
</template>

<script setup>
defineProps({
  image: {
    type: String,
    required: true
  },
  captionLabel: {
    type: String,
    default: 'PHOTO:'
  },
  caption: {
    type: String,
    required: true
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
  dateline: {
    type: String,
    default: ''
  },
  paragraphs: {
    type: Array,
    required: true
  },
  link: {
    type: String,
    default: ''
  }
})
</script>
