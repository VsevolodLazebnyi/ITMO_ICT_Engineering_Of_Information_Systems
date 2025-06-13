from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from dotenv import load_dotenv
from .data_processing import preprocess_data, prepare_features_target, train_and_predict_model, dataframe_to_bytes, \
    bytes_to_dataframe
from .database import save_processed_data, get_processed_data, list_processed_countries, init_db
from .models import PredictionRequest, UploadResponse, PredictionResponse, PredictionData, CountryData
import pandas as pd

# Загрузка переменных окружения
load_dotenv()

app = FastAPI(
    title="World Data Analytics API",
    description="API для анализа и прогнозирования мировых показателей",
    version="1.0.0"
)

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Конфигурация
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Инициализация базы данных
init_db()


@app.post("/upload", response_model=UploadResponse, tags=["Data Management"])
async def upload_file(file: UploadFile = File(...)):
    try:
        # Сохранение файла
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Предобработка данных
        processed_df = preprocess_data(file_path)

        # Извлечение кода страны
        country_code = "UNK"
        if 'Country Code' in processed_df.columns:
            # Берем первый попавшийся код страны из данных
            country_code = processed_df['Country Code'].iloc[0]
        elif 'country_code' in processed_df.columns:
            country_code = processed_df['country_code'].iloc[0]

        # Кэширование
        data_bytes = dataframe_to_bytes(processed_df)
        save_processed_data(filename, country_code, data_bytes)

        return UploadResponse(
            file_id=filename,
            country_code=country_code,
            message="File processed and cached successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/countries", response_model=list[CountryData], tags=["Data Management"])
async def list_countries():
    try:
        return list_processed_countries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Analytics"])
async def predict_data(request: PredictionRequest):
    try:
        # Поиск файла по коду страны
        countries = list_processed_countries()
        filename = None

        for country in countries:
            if country["country_code"] == request.country_code:
                filename = country["filename"]
                break

        if not filename:
            raise HTTPException(status_code=404, detail="Country data not found")

        # Получение данных
        data_bytes = get_processed_data(filename)
        if not data_bytes:
            raise HTTPException(status_code=404, detail="Processed data not found")

        df = bytes_to_dataframe(data_bytes)

        # Фильтрация по стране
        country_df = df[df['Country Code'] == request.country_code]
        if country_df.empty:
            if 'country_code' in df.columns:
                country_df = df[df['country_code'] == request.country_code]
        if country_df.empty:
            raise HTTPException(status_code=404, detail=f"Country data for {request.country_code} not found in dataset")

        country_name = country_df['Country Name'].iloc[
            0] if 'Country Name' in country_df.columns else request.country_code

        # Подготовка данных
        X_all, y_all, X_future = prepare_features_target(
            df=country_df,
            target_series_name=request.target_series,
            exclude_series=request.exclude_series,
            prediction_years_count=request.years_count
        )

        # Прогнозирование
        results = train_and_predict_model(
            X_all=X_all,
            y_all=y_all,
            X_future=X_future
        )

        # Форматирование ответа
        plot_df = results['plot_data']

        historical_data = []
        for _, row in plot_df.iterrows():
            if not pd.isna(row['Historical']):
                historical_data.append(
                    PredictionData(
                        year=int(row['Year']),
                        value=row['Historical']
                    )
                )

        forecast_data = []
        for _, row in plot_df.iterrows():
            if not pd.isna(row['Median']):
                forecast_data.append(
                    PredictionData(
                        year=int(row['Year']),
                        median=row['Median'],
                        lower=row['Lower'],
                        upper=row['Upper']
                    )
                )

        return PredictionResponse(
            historical=historical_data,
            forecast=forecast_data,
            country=country_name,
            target=request.target_series
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/series/{country_code}", response_model=list[str], tags=["Data Management"])
async def list_series(country_code: str):
    try:
        # Поиск файла по коду страны
        countries = list_processed_countries()
        filename = None
        for country in countries:
            if country["country_code"] == country_code:
                filename = country["filename"]
                break

        if not filename:
            raise HTTPException(status_code=404, detail="Country data not found")

        # Получение данных
        data_bytes = get_processed_data(filename)
        if not data_bytes:
            raise HTTPException(status_code=404, detail="Processed data not found")

        df = bytes_to_dataframe(data_bytes)
        # Фильтрация по стране
        country_df = df[df['Country Code'] == country_code]
        if country_df.empty and 'country_code' in df.columns:
            country_df = df[df['country_code'] == country_code]

        if country_df.empty:
            raise HTTPException(status_code=404, detail=f"Country data for {country_code} not found in dataset")

        # Извлечение уникальных показателей (Series Name)
        if 'Series Name' in country_df.columns:
            series_list = country_df['Series Name'].unique().tolist()
            return series_list
        else:
            raise HTTPException(status_code=404, detail="Series Name column not found in dataset")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)