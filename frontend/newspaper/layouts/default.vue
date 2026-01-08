<template>
  <div class="newspaper">
    <Masthead 
      :volume="'CLXXIII'"
      :issue="'59,847'"
      :date="formattedDate"
      :price="'$3.00'"
    />
    <main class="main-content">
      <slot />
    </main>
  </div>
</template>

<script setup>
const formattedDate = computed(() => {
  const now = new Date()
  const options = { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  }
  return now.toLocaleDateString('en-US', options).toUpperCase()
})
</script>
