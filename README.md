# Разработка приложения для работы с большими данными "BigDataCountryPredictor"
Дисциплина: "Проектирование информационных систем" <br>
Состав команды: Шабарина М.А, Махоткина Е.Д, Терёшкина Н.А, Гусейнова М.Э, Лазебный В.В

## 📝 Описание: 
BigDataCountryPredictor — это веб-приложение для работы с большими данными о странах мира на основе статистики Всемирного банка. Пользователь может выбрать страну и получить визуализацию ключевых социально-экономических параметров, а также прогноз на следующие 5 лет по выбранному индикатору.

Проект предоставляет следующий функционал:
- Загрузка Big Data с данными по странам
- Предобработка данных: очистка, нормализация, заполнение пропусков
- Обучение модели для прогноза значений на 5 лет вперёд
- Визуализация текущих и предсказанных данных по выбранной стране
- Веб-интерфейс с дашбордом и возможностью выбора страны

Реализованная модель обучается на исторических данных и предсказывает значения одного из выбранных параметров (например, GDP, уровень образования, численность населения) на 5 лет вперёд. Используются методы машинного обучения с кросс-валидацией и оценкой качества по метрикам RMSE/MAE.

## 💻 Технологии
- FastAPI
- SQL (PostgreSQL)
- Vue.js
- Pandas + Scikit-learn
- World Bank DataBank (источник данных)


## 🏛️ Архитектура проекта
```
├── BigDataApplication
│   ├── data
│   ├── data_preprocessing.py
│   ├── model_prediction.py
│   ├── preprocessing.ipynb
│   └── requirements.txt
├── README.md
├── backend
│   ├── app
│   │   ├── data_processing.py
│   │   ├── database.py
│   │   ├── main.py
│   │   └── models.py
│   ├── preprocessed_cache
│   │   └── data_cache.db
│   └── requirements.txt
└── world_analyze_project
    ├── README.md
    ├── index.html
    ├── jsconfig.json
    ├── package-lock.json
    ├── package.json
    ├── public
    │   └── favicon.ico
    ├── src
    │   ├── App.vue
    │   ├── assets
    │   │   ├── base.css
    │   │   ├── images
    │   │   │   └── logo.png
    │   │   ├── logo.svg
    │   │   └── main.css
    │   ├── components
    │   │   ├── CountrySelector.vue
    │   │   ├── FileUpload.vue
    │   │   ├── Footer.vue
    │   │   ├── ForecastChart.vue
    │   │   ├── Header.vue
    │   │   └── SeriesSelector.vue
    │   ├── main.js
    │   ├── pages
    │   │   └── Main.vue
    │   ├── router
    │   │   └── index.js
    │   ├── store
    │   │   └── index.js
    │   └── views
    │       └── Dashboard.vue
    └── vite.config.js
```
