<template>
  <header class="masthead">
    <div class="masthead-row">
      <div class="masthead-left">
        <div class="motto-box">"Today's Practice<br>Tomorrow's Trends"</div>
      </div>
      <div class="masthead-center">
        <h1 class="newspaper-title">The Artificial Intelligence Times</h1>
      </div>
      <div class="masthead-right">
        <div class="edition-box">Late Edition</div>
        <div>Today, partly cloudy, high 58.<br>Tomorrow, sunny, high 62.<br>Weather map: Page B14</div>
      </div>
    </div>
    <div class="masthead-bottom">
      <div>VOL. {{ volume }} ... No. {{ issue }}</div>
      <div>NEW YORK, {{ date }}</div>
      <div>{{ price }}</div>
    </div>
  </header>
</template>

<script setup>
defineProps({
  volume: {
    type: String,
    default: 'CLXXIII'
  },
  issue: {
    type: String,
    default: '59,847'
  },
  date: {
    type: String,
    required: true
  },
  price: {
    type: String,
    default: '$3.00'
  }
})
</script>
