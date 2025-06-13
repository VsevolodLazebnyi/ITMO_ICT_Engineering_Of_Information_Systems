<template>
  <div class="series-selector card">
    <h2>Select Series</h2>
    <div class="search-container">
      <input
          id="series-search"
          type="text"
          v-model="searchSeries"
          placeholder="Type to search..."
          @focus="showDropdown = true"
          @blur="hideDropdown"
          @input="filterSeries"
          @keydown.enter="selectFirstFiltered"
      />
      <div class="dropdown" v-if="showDropdown && filteredSeries.length">
        <div
            class="dropdown-item"
            v-for="series in filteredSeries"
            :key="series"
            @click="selectSeries(series)"
        >
          {{ series }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState, mapMutations } from 'vuex'

export default {
  name: 'SeriesSelector',
  data() {
    return {
      searchSeries: '',
      showDropdown: false,
      filteredSeries: []
    }
  },
  computed: {
    ...mapState(['seriesList']),
    selectedSeries: {
      get() {
        return this.$store.state.selectedSeries
      },
      set(value) {
        this.setSelectedSeries(value)
      }
    }
  },
  methods: {
    ...mapMutations(['setSelectedSeries']),
    filterSeries() {
      if (this.searchSeries) {
        this.filteredSeries = this.seriesList.filter(series =>
            series.toLowerCase().includes(this.searchSeries.toLowerCase())
        )
      } else {
        this.filteredSeries = [...this.seriesList]
      }
    },
    selectSeries(series) {
      this.setSelectedSeries(series)
      this.searchSeries = series
      this.showDropdown = false
      this.$store.dispatch('getForecast')
    },
    selectFirstFiltered() {
      if (this.filteredSeries.length > 0) {
        this.selectSeries(this.filteredSeries[0])
      }
    },
    hideDropdown() {
      setTimeout(() => {
        this.showDropdown = false
      }, 200)
    }
  },
  watch: {
    selectedSeries(newVal) {
      if (newVal) {
        this.searchSeries = newVal
      } else {
        this.searchSeries = ''
      }
    },
    seriesList() {
      this.filterSeries()
    }
  }
}
</script>

<style scoped>
.series-selector {
  text-align: center;
  min-width: 450px;
  padding: 1rem 1.5rem;
}

.card {
  background-color: #f3f4f6;
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
}

h2 {
  font-size: 1rem;
  color: #6b7280;
  margin-bottom: 0.75rem;
}

.search-container {
  position: relative;
}

input {
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  border: 1px solid #3771f3;
  width: 100%;
  font-size: 0.9rem;
  outline: none;
  transition: border-color 0.3s ease;
  background-color: white;
}

input:focus {
  border-color: #3771f3;
  box-shadow: 0 0 0 2px rgba(55, 113, 243, 0.2);
}

.dropdown {
  position: absolute;
  top: calc(100% + 0.5rem);
  left: 0;
  right: 0;
  background: white;
  border: 1px solid #3771f3;
  border-radius: 6px;
  max-height: 200px;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  z-index: 10;
}

.dropdown-item {
  padding: 0.5rem 0.75rem;
  cursor: pointer;
  transition: background 0.2s ease;
  font-size: 0.9rem;
  text-align: left;
}

.dropdown-item:hover {
  background-color: rgba(55, 113, 243, 0.1);
  color: #3771f3;
}
</style>