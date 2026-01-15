// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: false },

  modules: ['@nuxt/content'],

  ssr: true,

  experimental: {
    payloadExtraction: false
  },
  
  app: {
    baseURL: '/',
    head: {
      title: 'The Artificial Intelligence Times',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'The Newspaper of Record for Artificial Intelligence' }
      ],
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        { 
          rel: 'stylesheet', 
          href: 'https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=UnifrakturMaguntia&display=swap' 
        }
      ]
    }
  },

  css: ['~/assets/css/newspaper.css'],

  // Development server configuration
  devServer: {
    port: 12000
  },

  // Static site generation
  nitro: {
    prerender: {
      crawlLinks: true,
      routes: [
        '/',
        '/articles/breakthrough-model-achieves-new-benchmark-in-scientific-reasoning',
        '/articles/senate-opens-sweeping-inquiry-into-ai-safety-standards',
        '/articles/china-unveils-50-billion-ai-development-plan',
        '/articles/ai-system-identifies-new-antibiotic-candidates',
        '/articles/european-regulators-propose-strict-new-ai-rules',
        '/articles/universities-overhaul-curricula-to-meet-surging-ai-demand',
        '/articles/labor-unions-call-for-ai-worker-protections',
        '/articles/tech-giants-report-record-quarterly-earnings',
        '/articles/startups-race-to-build-specialized-ai-chips',
        '/articles/schools-grapple-with-ai-in-classrooms',
        '/articles/global-ai-summit-set-for-march-in-geneva',
        '/articles/wall-street-debates-ai-valuations',
        '/articles/supreme-court-to-hear-ai-copyright-case',
        '/articles/pentagon-boosts-ai-budget-by-40-percent',
        '/articles/moma-exhibits-ai-generated-art',
        '/articles/hollywood-writers-reach-ai-deal',
        '/articles/openai-names-new-board-members',
        '/articles/japan-and-us-partner-on-robotics-research-lab',
        '/articles/britain-plans-ai-safety-institute',
        '/articles/nvidia-stock-hits-new-record-high',
        '/articles/microsoft-expands-azure-ai-services'
      ]
    }
  },

  compatibilityDate: '2024-01-01'
})
