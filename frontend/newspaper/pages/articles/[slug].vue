<template>
  <div class="article-page">
    <div class="article-container">
      <div class="article-header">
        <div class="section-label">{{ article?.section || 'News' }}</div>
        <h1 class="article-headline">{{ article?.headline || 'Article Not Found' }}</h1>
        <div v-if="article?.subhead" class="article-subhead">{{ article.subhead }}</div>
        <div v-if="article?.byline" class="article-byline">By {{ article.byline }}</div>
        <div v-if="article?.dateline" class="article-dateline">{{ article.dateline }}</div>
      </div>

      <div v-if="article?.image" class="article-image">
        <img :src="article.image" :alt="article.headline" />
        <div v-if="article?.caption" class="caption">
          <span v-if="article?.captionLabel" class="caption-label">{{ article.captionLabel }}</span>
          {{ article.caption }}
        </div>
      </div>

      <div class="article-body">
        <p v-for="(paragraph, index) in article?.paragraphs || []" :key="index">
          {{ paragraph }}
        </p>
        <p v-if="article?.text && !article?.paragraphs">
          {{ article.text }}
        </p>
        <p v-if="!article">
          This article could not be found. Please return to the <NuxtLink to="/">homepage</NuxtLink>.
        </p>
      </div>

      <div class="article-footer">
        <NuxtLink to="/" class="back-link">← Back to Homepage</NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup>
import { heroArticle, featuredArticles, secondaryMain, sidebarArticles } from '~/data/articles.js'

const route = useRoute()
const slug = route.params.slug

// Find the article by matching the slug from the link
const findArticle = () => {
  // Check hero article
  if (heroArticle.link === `/articles/${slug}`) {
    return heroArticle
  }

  // Check featured articles
  for (const article of featuredArticles) {
    if (article.link === `/articles/${slug}`) {
      return article
    }
  }

  // Check secondary main
  if (secondaryMain.link === `/articles/${slug}`) {
    return secondaryMain
  }

  // Check sidebar articles
  for (const column of sidebarArticles) {
    for (const article of column) {
      if (article.link === `/articles/${slug}`) {
        return article
      }
    }
  }

  return null
}

const article = findArticle()

// Set page metadata
useHead({
  title: article ? `${article.headline} - The Artificial Intelligence Times` : 'Article Not Found - The Artificial Intelligence Times',
  meta: [
    { name: 'description', content: article?.text || article?.subhead || '' }
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

.article-body p {
  margin-bottom: 20px;
  text-align: justify;
}

.article-body p:first-letter {
  font-size: 3em;
  line-height: 0.8;
  float: left;
  margin: 5px 8px 0 0;
  font-weight: 700;
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
