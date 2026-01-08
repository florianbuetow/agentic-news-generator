<template>
  <section class="briefs-section">
    <div class="briefs-grid">
      <div 
        v-for="(column, index) in columns" 
        :key="index" 
        class="brief-column"
      >
        <div class="brief-section-label">{{ column.section }}</div>
        <BriefItem
          v-for="(brief, briefIndex) in column.items"
          :key="briefIndex"
          :headline="brief.headline"
          :text="brief.text"
          :link="brief.link"
        />
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  columns: {
    type: Array,
    required: true
  }
})
</script>
