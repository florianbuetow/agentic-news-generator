<template>
  <div class="article-page">
    <div v-if="error" class="article-container">
      <div class="article-header">
        <h1 class="article-headline">Article Not Found</h1>
      </div>
      <div class="article-body">
        <p>
          This article could not be found. Please return to the <NuxtLink to="/">homepage</NuxtLink>.
        </p>
      </div>
    </div>
    <div v-else-if="article" class="article-container">
      <div class="article-header">
        <div class="section-label">{{ article.tags?.[0] || 'News' }}</div>
        <h1 class="article-headline">{{ article.title }}</h1>
        <div v-if="article.subtitle" class="article-subhead">{{ article.subtitle }}</div>
        <div v-if="article.author" class="article-byline">By {{ article.author }}</div>
        <div v-if="article.date" class="article-dateline">{{ formatDate(article.date) }}</div>
      </div>

      <div v-if="articleImage" class="article-image">
        <img :src="articleImage" :alt="article.title" />
        <div v-if="article.summary" class="caption">
          <span class="caption-label">{{ article.tags?.[0]?.toUpperCase() }}:</span>
          {{ article.summary }}
        </div>
      </div>

      <div class="article-body">
        <ContentRenderer :value="article" />
      </div>

      <div class="article-footer">
        <NuxtLink to="/" class="back-link">← Back to Homepage</NuxtLink>
      </div>
    </div>
    <div v-else class="article-container">
      <p>Loading...</p>
    </div>
  </div>
</template>

<script setup>
const route = useRoute()
const slug = route.params.slug

// Query the article by slug
const { data: article, error } = await useAsyncData(`article-${slug}`, () =>
  queryCollection('articles').where('slug', '=', slug).first()
)

// Extract image from article
const articleImage = computed(() => {
  if (!article.value) return null

  // Try frontmatter first
  if (article.value.image) return article.value.image

  // Extract from markdown body
  const imgRegex = /!\[.*?\]\((https?:\/\/[^\)]+)\)/
  const match = article.value.body?.raw?.match?.(imgRegex)
  return match ? match[1] : null
})

// Format date helper
const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

// Set page metadata
useHead({
  title: article.value ? `${article.value.title} - The Artificial Intelligence Times` : 'Article Not Found - The Artificial Intelligence Times',
  meta: [
    { name: 'description', content: article.value?.summary || article.value?.subtitle || '' }
  ]
})
</script>

<style scoped>
.article-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Libre Baskerville', serif;
}

.article-container {
  background: white;
}

.article-header {
  margin-bottom: 30px;
  border-bottom: 2px solid #000;
  padding-bottom: 20px;
}

.section-label {
  font-family: 'Playfair Display', serif;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 10px;
}

.article-headline {
  font-family: 'Playfair Display', serif;
  font-size: 42px;
  font-weight: 800;
  line-height: 1.1;
  margin: 0 0 15px 0;
}

.article-subhead {
  font-size: 20px;
  font-style: italic;
  line-height: 1.4;
  margin-bottom: 15px;
  color: #333;
}

.article-byline {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 5px;
}

.article-dateline {
  font-size: 12px;
  font-style: italic;
  color: #666;
}

.article-image {
  margin-bottom: 30px;
}

.article-image img {
  width: 100%;
  height: auto;
  display: block;
}

.caption {
  font-size: 12px;
  line-height: 1.4;
  padding: 10px 0;
  border-top: 1px solid #ddd;
  color: #666;
}

.caption-label {
  font-weight: 700;
  margin-right: 5px;
}

.article-body {
  font-size: 18px;
  line-height: 1.8;
}

/* Style the rendered markdown content */
.article-body :deep(p) {
  margin-bottom: 20px;
  text-align: justify;
}

.article-body :deep(p:first-of-type:first-letter) {
  font-size: 3em;
  line-height: 0.8;
  float: left;
  margin: 5px 8px 0 0;
  font-weight: 700;
}

.article-body :deep(h2),
.article-body :deep(h3) {
  font-family: 'Playfair Display', serif;
  font-weight: 700;
  margin: 30px 0 15px 0;
}

.article-body :deep(h2) {
  font-size: 28px;
}

.article-body :deep(h3) {
  font-size: 22px;
}

.article-body :deep(blockquote) {
  border-left: 4px solid #000;
  padding-left: 20px;
  margin: 20px 0;
  font-style: italic;
  color: #333;
}

.article-body :deep(ul),
.article-body :deep(ol) {
  margin: 20px 0;
  padding-left: 40px;
}

.article-body :deep(li) {
  margin-bottom: 10px;
}

.article-body :deep(code) {
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 16px;
}

.article-body :deep(pre) {
  background: #f5f5f5;
  padding: 20px;
  border-radius: 5px;
  overflow-x: auto;
  margin: 20px 0;
}

.article-body :deep(pre code) {
  background: none;
  padding: 0;
}

.article-body :deep(a) {
  color: #000;
  text-decoration: underline;
}

.article-body :deep(a:hover) {
  color: #333;
}

.article-body :deep(img) {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 20px auto;
}

.article-footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #ddd;
}

.back-link {
  font-size: 14px;
  text-decoration: none;
  color: #000;
  font-weight: 700;
}

.back-link:hover {
  text-decoration: underline;
}

@media (max-width: 768px) {
  .article-headline {
    font-size: 32px;
  }

  .article-body {
    font-size: 16px;
  }
}
</style>
