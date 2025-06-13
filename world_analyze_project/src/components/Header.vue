<template>
  <header class="header">
    <div class="logo-container">
      <img :src="logo" alt="Логотип" class="logo" />
    </div>
  </header>
</template>

<script setup>
import logo from '@/assets/images/logo.png'
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #1F2937;
  padding: 3%;
  position: relative;
}

.logo-container {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.logo {
  height: 50px;
  object-fit: contain;
}
</style>