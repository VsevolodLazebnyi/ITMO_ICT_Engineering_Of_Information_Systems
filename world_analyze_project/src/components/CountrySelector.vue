<template>
  <div class="country-selector card">
    <h2>Select Country</h2>
    <div class="search-container">
      <input
          id="country-search"
          type="text"
          v-model="searchCountry"
          placeholder="Type to search..."
          @focus="showDropdown = true"
          @blur="hideDropdown"
          @input="filterCountries"
          @keydown.enter="selectFirstFiltered"
      />
      <div class="dropdown" v-if="showDropdown && filteredCountries.length">
        <div
            class="dropdown-item"
            v-for="country in filteredCountries"
            :key="country.country_code"
            @click="selectCountry(country.country_code)"
        >
          {{ country.country_code }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState, mapMutations } from 'vuex'

export default {
  name: 'CountrySelector',
  data() {
    return {
      searchCountry: '',
      showDropdown: false,
      filteredCountries: []
    }
  },
  computed: {
    ...mapState(['countries']),
    selectedCountry: {
      get() {
        return this.$store.state.selectedCountry
      },
      set(value) {
        this.setSelectedCountry(value)
      }
    }
  },
  methods: {
    ...mapMutations(['setSelectedCountry']),
    filterCountries() {
      if (this.searchCountry) {
        this.filteredCountries = this.countries.filter(country =>
            country.country_code.toLowerCase().includes(this.searchCountry.toLowerCase())
        )
      } else {
        this.filteredCountries = [...this.countries]
      }
    },
    selectCountry(code) {
      this.setSelectedCountry(code)
      this.searchCountry = code
      this.showDropdown = false
      this.$store.dispatch('loadSeries')
      this.$store.dispatch('getForecast')
    },
    selectFirstFiltered() {
      if (this.filteredCountries.length > 0) {
        this.selectCountry(this.filteredCountries[0].country_code)
      }
    },
    hideDropdown() {
      setTimeout(() => {
        this.showDropdown = false
      }, 200)
    }
  },
  watch: {
    selectedCountry(newVal) {
      if (newVal) {
        this.searchCountry = newVal
      } else {
        this.searchCountry = ''
      }
    }
  },
  mounted() {
    this.$store.dispatch('loadCountries')
    this.filterCountries()
  }
}
</script>

<style scoped>
.country-selector {
  text-align: center;
  min-width: 200px;
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