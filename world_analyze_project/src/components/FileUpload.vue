<template>
  <div class="file-upload">
    <input type="file" ref="fileInput" @change="handleFileUpload" accept=".csv" />
    <button @click="triggerFileInput">Upload CSV</button>
    <p v-if="uploadMessage">{{ uploadMessage }}</p>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'FileUpload',
  data() {
    return {
      uploadMessage: ''
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },
    async handleFileUpload(event) {
      const file = event.target.files[0]
      if (!file) return

      const formData = new FormData()
      formData.append('file', file)

      try {
        const response = await axios.post('http://localhost:8000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        this.uploadMessage = response.data.message
        this.$store.dispatch('loadCountries')
        setTimeout(() => {
          this.uploadMessage = ''
        }, 3000)
      } catch (error) {
        this.uploadMessage = 'Error uploading file: ' + error.message
      }
    }
  }
}
</script>

<style scoped>
.file-upload {
  margin: 1rem 0;
}

input[type="file"] {
  display: none;
}

button {
  background-color: #3771f3;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

button:hover {
  background-color: #2a5bc7;
}

p {
  margin-top: 0.5rem;
  color: #28a745;
}
</style>