<template>
  <div>
    <!-- Hero Section -->
    <HeroSection
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
        <div class="secondary-main">
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
          />
        </div>
      </div>
    </section>

    <!-- Briefs Section -->
    <BriefsSection :columns="briefsColumns" />
  </div>
</template>

<script setup>
import { 
  heroArticle, 
  featuredArticles, 
  secondaryMain, 
  sidebarArticles, 
  briefsColumns 
} from '~/data/articles.js'
</script>
