<template>
  <div class="dashboard">
    <!-- <h1>World Data Analytics Dashboard</h1> -->

    <div class="controls">
      <div class="selectors">
        <CountrySelector />
        <SeriesSelector />
      </div>
    </div>

    <ForecastChart />
  </div>
</template>

<script>
import CountrySelector from '@/components/CountrySelector.vue'
import SeriesSelector from '@/components/SeriesSelector.vue'
import ForecastChart from '@/components/ForecastChart.vue'

export default {
  name: 'Dashboard',
  components: {
    CountrySelector,
    SeriesSelector,
    ForecastChart
  }
}
</script>

<style scoped>
.dashboard {
  padding: 2rem;
  flex: 1;
}

h1 {
  text-align: center;
  margin-bottom: 2rem;
  color: #1F2937;
}

.controls {
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.selectors {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 1.5rem;
}
</style>