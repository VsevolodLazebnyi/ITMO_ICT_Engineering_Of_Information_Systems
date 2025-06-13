<template>
  <div class="forecast-chart">
    <div v-if="isLoading" class="loading">Loading...</div>

    <div v-else-if="forecastData">
      <div class="metrics">
        <div class="card">
          <h2>Growth</h2>
          <p>{{ calculateGrowth() }}</p>
        </div>
        <div class="card">
          <h2>Average</h2>
          <p>{{ calculateAverage() }}</p>
        </div>
        <div class="card">
          <h2>Forecast</h2>
          <p>{{ getForecastValue() }}</p>
        </div>
      </div>

      <div class="chart-container">
        <canvas ref="chartCanvas"></canvas>
      </div>
    </div>

    <div v-else class="no-data">
      <p>Select a country to see the forecast</p>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted } from 'vue'
import { Chart, registerables } from 'chart.js'
import { mapState } from 'vuex'

Chart.register(...registerables)

export default {
  name: 'ForecastChart',
  setup() {
    const chartCanvas = ref(null)
    let chartInstance = null

    return {
      chartCanvas,
      chartInstance
    }
  },
  computed: {
    ...mapState(['forecastData', 'isLoading', 'selectedSeries'])
  },
  watch: {
    forecastData(newData) {
      if (newData) {
        this.renderChart()
      }
    }
  },
  methods: {
    renderChart() {
      if (this.chartInstance) {
        this.chartInstance.destroy()
      }

      if (!this.forecastData || !this.chartCanvas) return

      const ctx = this.chartCanvas.getContext('2d')

      // Prepare data
      const historical = this.forecastData.historical
      const forecast = this.forecastData.forecast

      const labels = [
        ...historical.map(d => d.year),
        ...forecast.map(d => d.year)
      ]

      const historicalData = [
        ...historical.map(d => d.value),
        ...forecast.map(() => null)
      ]

      const forecastData = [
        ...historical.map(() => null),
        ...forecast.map(d => d.median)
      ]

      const lowerBound = [
        ...historical.map(() => null),
        ...forecast.map(d => d.lower)
      ]

      const upperBound = [
        ...historical.map(() => null),
        ...forecast.map(d => d.upper)
      ]

      this.chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Historical ' + this.selectedSeries,
              data: historicalData,
              borderColor: '#3771f3',
              backgroundColor: 'rgba(55, 113, 243, 0.2)',
              fill: true,
              tension: 0.3
            },
            {
              label: 'Forecast',
              data: forecastData,
              borderColor: '#ff6384',
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              fill: false,
              borderDash: [5, 5]
            },
            {
              label: 'Lower Bound',
              data: lowerBound,
              borderColor: 'rgba(255, 99, 132, 0.3)',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              fill: '-1',
              pointRadius: 0,
              borderWidth: 1
            },
            {
              label: 'Upper Bound',
              data: upperBound,
              borderColor: 'rgba(255, 99, 132, 0.3)',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              fill: '-1',
              pointRadius: 0,
              borderWidth: 1
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'top'
            },
            title: {
              display: true,
              text: `${this.selectedSeries} Forecast for ${this.forecastData.country}`
            },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Year'
              }
            },
            y: {
              title: {
                display: true,
                text: this.selectedSeries
              }
            }
          }
        }
      })
    },
    calculateGrowth() {
      if (!this.forecastData || this.forecastData.historical.length < 2) return 'N/A'

      const historical = this.forecastData.historical
      const last = historical[historical.length - 1].value
      const prev = historical[historical.length - 2].value

      const growth = ((last - prev) / prev) * 100
      return `${growth > 0 ? '+' : ''}${growth.toFixed(1)}%`
    },
    calculateAverage() {
      if (!this.forecastData) return 'N/A'

      const values = this.forecastData.historical.map(d => d.value)
      const avg = values.reduce((a, b) => a + b, 0) / values.length

      // Format large numbers
      if (avg > 1e9) return `${(avg / 1e9).toFixed(1)}B`
      if (avg > 1e6) return `${(avg / 1e6).toFixed(1)}M`
      if (avg > 1e3) return `${(avg / 1e3).toFixed(1)}K`

      return avg.toFixed(1)
    },
    getForecastValue() {
      if (!this.forecastData || this.forecastData.forecast.length === 0) return 'N/A'

      const lastForecast = this.forecastData.forecast[this.forecastData.forecast.length - 1]
      const value = lastForecast.median
      const year = lastForecast.year

      // Format large numbers
      let formattedValue = value
      if (value > 1e9) formattedValue = `${(value / 1e9).toFixed(1)}B`
      else if (value > 1e6) formattedValue = `${(value / 1e6).toFixed(1)}M`
      else if (value > 1e3) formattedValue = `${(value / 1e3).toFixed(1)}K`
      else formattedValue = value.toFixed(1)

      return `${formattedValue} by ${year}`
    }
  },
  mounted() {
    if (this.forecastData) {
      this.renderChart()
    }
  }
}
</script>

<style scoped>
.forecast-chart {
  margin-top: 2rem;
}

.loading, .no-data {
  text-align: center;
  padding: 2rem;
  font-size: 1.2rem;
  color: #666;
}

.metrics {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 2rem;
}

.card {
  background-color: #f3f4f6;
  padding: 1rem 2rem;
  border-radius: 10px;
  text-align: center;
  min-width: 150px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
}

.card h2 {
  font-size: 1rem;
  color: #6b7280;
  margin-bottom: 0.5rem;
}

.card p {
  font-size: 1.5rem;
  font-weight: bold;
}

.chart-container {
  max-width: 800px;
  margin: 0 auto;
  height: 500px;
}
</style>