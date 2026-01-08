<template>
  <div v-if="error">
    <h1>Error Loading Content</h1>
    <p>{{ error }}</p>
  </div>
  <div v-else-if="!layout || !allArticles">
    <p>Loading...</p>
  </div>
  <div v-else>
    <!-- Hero Section -->
    <HeroSection
      v-if="heroArticle"
      :image="heroArticle.image"
      :caption-label="heroArticle.captionLabel"
      :caption="heroArticle.caption"
      :headline="heroArticle.headline"
      :subhead="heroArticle.subhead"
      :byline="heroArticle.byline"
      :dateline="heroArticle.dateline"
      :paragraphs="heroArticle.paragraphs"
      :link="heroArticle.link"
    />

    <!-- Main Articles Grid -->
    <section class="articles-section">
      <h3 class="section-title">Top Stories</h3>
      <div class="articles-grid">
        <ArticleCard
          v-for="(article, index) in featuredArticles"
          :key="index"
          :image="article.image"
          :section="article.section"
          :headline="article.headline"
          :subhead="article.subhead"
          :byline="article.byline"
          :text="article.text"
          :link="article.link"
          :size="article.size || 'normal'"
        />
      </div>
    </section>

    <!-- Secondary Section -->
    <section class="secondary-section">
      <div class="secondary-grid">
        <!-- Main Secondary Article -->
        <div v-if="secondaryMain" class="secondary-main">
          <img
            class="secondary-main-image"
            :src="secondaryMain.image"
            :alt="secondaryMain.headline"
          >
          <h3 class="secondary-main-headline">{{ secondaryMain.headline }}</h3>
          <p class="article-card-byline">By {{ secondaryMain.byline }}</p>
          <div class="secondary-main-text">
            <p v-for="(paragraph, index) in secondaryMain.paragraphs" :key="index">
              <span v-if="index === 0 && secondaryMain.dateline" class="dateline">
                {{ secondaryMain.dateline }} —
              </span>
              {{ paragraph }}
            </p>
          </div>
          <NuxtLink :to="secondaryMain.link" class="read-more">Continue Reading →</NuxtLink>
        </div>

        <!-- Sidebar Columns -->
        <div
          v-for="(column, colIndex) in sidebarArticles"
          :key="colIndex"
          class="secondary-sidebar"
        >
          <SidebarArticle
            v-for="(article, articleIndex) in column"
            :key="articleIndex"
            :headline="article.headline"
            :byline="article.byline"
            :text="article.text"
            :link="article.link"
          />
        </div>
      </div>
    </section>

    <!-- Briefs Section -->
    <BriefsSection :columns="briefsColumns" />
  </div>
</template>

<script setup>
// Query layout configuration
const { data: layout, error: layoutError } = await useAsyncData('layout', () =>
  queryCollection('layout').first()
)

// Query all articles
const { data: allArticles, error: articlesError } = await useAsyncData('articles', () =>
  queryCollection('articles').all()
)

const error = ref(null)

if (layoutError.value) {
  error.value = `Failed to load layout: ${layoutError.value}`
} else if (articlesError.value) {
  error.value = `Failed to load articles: ${articlesError.value}`
}

// Helper function to find article by slug
const findArticleBySlug = (slug) => {
  const articles = Array.isArray(allArticles.value)
    ? allArticles.value
    : allArticles.value?.data || allArticles.value?.items || []

  return articles.find(a => a.slug === slug || a._path?.endsWith(`/${slug}`))
}

// Helper function to extract image from markdown
const extractImage = (article) => {
  // Try to get image from frontmatter first
  if (article.image) return article.image

  // Extract from markdown body
  const imgRegex = /!\[.*?\]\((https?:\/\/[^\)]+)\)/
  const match = article.body?.raw?.match?.(imgRegex)
  return match ? match[1] : null
}

// Helper function to extract paragraphs from markdown body
const extractParagraphs = (article, count = null) => {
  if (!article.body?.children) return []

  const paragraphs = []
  for (const node of article.body.children) {
    if (node.tag === 'p' && node.children) {
      // Extract text from paragraph node
      const text = node.children
        .map(child => child.value || child.children?.[0]?.value || '')
        .join('')
        .trim()
      if (text) paragraphs.push(text)
    }
    if (count && paragraphs.length >= count) break
  }

  return paragraphs
}

// Helper function to format article for hero section
const formatHeroArticle = (article) => {
  if (!article) return null
  return {
    image: extractImage(article) || 'https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&q=80',
    captionLabel: article.tags?.[0]?.toUpperCase() + ':' || 'NEWS:',
    caption: article.summary || '',
    headline: article.title,
    subhead: article.subtitle || '',
    byline: article.author || '',
    dateline: article.dateline || '',
    paragraphs: extractParagraphs(article, 2),
    link: `/articles/${article.slug}`
  }
}

// Helper function to format article for featured section
const formatFeaturedArticle = (article, isLarge = false) => {
  if (!article) return null
  const paragraphs = extractParagraphs(article, 1)
  return {
    image: extractImage(article),
    section: article.tags?.[0] || 'News',
    headline: article.title,
    subhead: isLarge ? article.subtitle : undefined,
    byline: article.author || '',
    text: paragraphs[0] || article.summary || '',
    link: `/articles/${article.slug}`,
    size: isLarge ? 'large' : undefined
  }
}

// Helper function to format article for secondary section
const formatSecondaryArticle = (article) => {
  if (!article) return null
  return {
    image: extractImage(article) || 'https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&q=80',
    headline: article.title,
    byline: article.author || '',
    dateline: article.dateline || '',
    paragraphs: extractParagraphs(article, 2),
    link: `/articles/${article.slug}`
  }
}

// Helper function to format article for sidebar
const formatSidebarArticle = (article) => {
  if (!article) return null
  return {
    headline: article.title,
    byline: article.author || '',
    text: article.summary || '',
    link: `/articles/${article.slug}`
  }
}

// Helper function to format article for brief
const formatBriefArticle = (article) => {
  if (!article) return null
  return {
    headline: article.title,
    text: article.summary || '',
    link: `/articles/${article.slug}`
  }
}

// Map articles to sections based on layout
const heroArticle = computed(() => {
  if (!layout.value?.meta?.hero) return null
  const article = findArticleBySlug(layout.value.meta.hero)
  return formatHeroArticle(article)
})

const featuredArticles = computed(() => {
  if (!layout.value?.meta?.featured) return []
  return layout.value.meta.featured.map((slug, index) => {
    const article = findArticleBySlug(slug)
    return formatFeaturedArticle(article, index === 0) // First one is large
  }).filter(Boolean)
})

const secondaryMain = computed(() => {
  if (!layout.value?.meta?.secondary) return null
  const article = findArticleBySlug(layout.value.meta.secondary)
  return formatSecondaryArticle(article)
})

const sidebarArticles = computed(() => {
  if (!layout.value?.meta?.sidebar) return []
  return layout.value.meta.sidebar.map(column =>
    column.map(slug => {
      const article = findArticleBySlug(slug)
      return formatSidebarArticle(article)
    }).filter(Boolean)
  )
})

const briefsColumns = computed(() => {
  if (!layout.value?.meta?.briefs) return []
  const sections = ['National', 'International', 'Business', 'Arts & Culture']
  return layout.value.meta.briefs.map((columnSlugs, index) => ({
    section: sections[index] || 'News',
    items: columnSlugs.map(slug => {
      const article = findArticleBySlug(slug)
      return formatBriefArticle(article)
    }).filter(Boolean)
  }))
})
</script>
