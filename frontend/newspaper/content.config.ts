import { defineContentConfig, defineCollection, z } from '@nuxt/content'

export default defineContentConfig({
  collections: {
    articles: defineCollection({
      type: 'page',
      source: '**/*.md',
      schema: z.object({
        title: z.string(),
        subtitle: z.string().optional(),
        slug: z.string(),
        author: z.string().optional(),
        date: z.string().optional(),
        dateline: z.string().optional(),
        tags: z.array(z.string()).optional(),
        summary: z.string().optional(),
        image: z.string().optional(),
      })
    }),
    layout: defineCollection({
      type: 'data',
      source: 'layout.json'
    })
  }
})
