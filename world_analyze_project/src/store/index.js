import { createStore } from 'vuex'
import axios from 'axios'

export default createStore({
    state: {
        countries: [],
        selectedCountry: null,
        seriesList: [],
        selectedSeries: "GDP (current US$)",
        forecastData: null,
        isLoading: false
    },
    mutations: {
        setCountries(state, countries) {
            state.countries = countries
        },
        setSelectedCountry(state, country) {
            state.selectedCountry = country
        },
        setSeriesList(state, series) {
            state.seriesList = series
        },
        setSelectedSeries(state, series) {
            state.selectedSeries = series
        },
        setForecastData(state, data) {
            state.forecastData = data
        },
        setLoading(state, isLoading) {
            state.isLoading = isLoading
        }
    },
    actions: {
        async loadCountries({ commit }) {
            try {
                const response = await axios.get('http://localhost:8000/countries')
                commit('setCountries', response.data)
            } catch (error) {
                console.error('Error loading countries:', error)
            }
        },
        async loadSeries({ commit, state }) {
            if (!state.selectedCountry) return
            try {
                const response = await axios.get(`http://localhost:8000/series/${state.selectedCountry}`)
                commit('setSeriesList', response.data)
                // Если selectedSeries не установлен или его нет в списке, выбираем первый из доступных
                if (!state.selectedSeries || !response.data.includes(state.selectedSeries)) {
                    commit('setSelectedSeries', response.data.length > 0 ? response.data[0] : '')
                }
            } catch (error) {
                console.error('Error loading series:', error)
            }
        },
        async getForecast({ commit, state }) {
            if (!state.selectedCountry || !state.selectedSeries) return
            commit('setLoading', true)
            try {
                const response = await axios.post('http://localhost:8000/predict', {
                    country_code: state.selectedCountry,
                    target_series: state.selectedSeries
                })
                commit('setForecastData', response.data)
            } catch (error) {
                console.error('Error getting forecast:', error)
            } finally {
                commit('setLoading', false)
            }
        }
    }
})