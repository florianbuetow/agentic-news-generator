# The Artificial Intelligence Times

A 1950s New York Times inspired newspaper template built with Nuxt 3.

## Features

- **Authentic newspaper masthead** with gothic typography
- **Responsive design** - works on desktop, tablet, and mobile
- **Modular components** - reusable ArticleCard, HeroSection, BriefsSection, etc.
- **Static site generation** - deploy anywhere
- **Content-driven** - articles authored as Markdown under `content/articles/`, arranged by `content/layout.json` via `@nuxt/content`

## Project Structure

```
newspaper/
├── app.vue                 # App entry point
├── nuxt.config.ts          # Nuxt configuration (dev port 12000)
├── content.config.ts       # @nuxt/content collections: articles + layout
├── package.json            # Dependencies
├── assets/
│   └── css/
│       └── newspaper.css   # Main stylesheet
├── components/
│   ├── Masthead.vue        # Newspaper header
│   ├── HeroSection.vue     # Featured article with image
│   ├── ArticleCard.vue     # Article card (normal/large)
│   ├── SidebarArticle.vue  # Compact sidebar article
│   ├── BriefItem.vue       # Single news brief
│   └── BriefsSection.vue   # Briefs grid section
├── content/
│   ├── articles/           # One Markdown file per article (YAML frontmatter + body)
│   └── layout.json         # Front-page layout by article slug
├── layouts/
│   └── default.vue         # Default page layout
└── pages/
    ├── index.vue           # Homepage (queries articles + layout collections)
    └── articles/[slug].vue # Individual article page
```

## Setup

Install dependencies:

```bash
npm install
```

## Development

Start the development server on `http://localhost:12000`:

```bash
npm run dev
```

## Production

### Generate Static Site

Generate a static version of the site:

```bash
npm run generate
```

The static files will be output to `.output/public/`. You can deploy this folder to any static hosting service (Netlify, Vercel, GitHub Pages, etc.).

### Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

## Customization

### Updating Content

Content is managed via `@nuxt/content` (see `content.config.ts`), with two collections:

- **`articles`** — one Markdown file per article under `content/articles/`, each with YAML frontmatter (`title`, `subtitle`, `slug`, `author`, `date`, `dateline`, `tags`, `summary`, `image`) followed by the article body.
- **`layout`** — `content/layout.json`, which arranges the front page by article slug:
  - `hero` - The main featured article
  - `featured` - Grid of top stories
  - `secondary` - Secondary featured article
  - `sidebar` - Sidebar article columns
  - `briefs` - News briefs at the bottom

### Adding New Pages

Create new `.vue` files in the `pages/` directory. Nuxt automatically creates routes based on the file structure.

Example: `pages/articles/[slug].vue` for individual article pages.

### Styling

The main stylesheet is in `assets/css/newspaper.css`. CSS variables and component-specific styles can be modified there.

## Components API

### ArticleCard

```vue
<ArticleCard
  image="/path/to/image.jpg"
  section="Technology"
  headline="Article Headline"
  subhead="Optional subheadline"
  byline="AUTHOR NAME"
  text="Article preview text..."
  link="/articles/slug"
  size="normal|large"
/>
```

### HeroSection

```vue
<HeroSection
  image="/path/to/image.jpg"
  caption-label="PHOTO:"
  caption="Image caption text"
  headline="Main Headline"
  subhead="Subheadline text"
  byline="AUTHOR NAME"
  dateline="NEW YORK"
  :paragraphs="['Paragraph 1', 'Paragraph 2']"
  link="/articles/slug"
/>
```

### SidebarArticle

```vue
<SidebarArticle
  headline="Article Headline"
  byline="AUTHOR NAME"
  text="Brief article text..."
/>
```

### BriefsSection

```vue
<BriefsSection
  :columns="[
    {
      section: 'National',
      items: [
        { headline: 'Brief Headline', text: 'Brief text' }
      ]
    }
  ]"
/>
```

## License

MIT
