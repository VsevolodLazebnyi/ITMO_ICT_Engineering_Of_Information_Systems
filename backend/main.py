from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from data_processing import preprocess_data_spark, prepare_features_target, train_and_predict_model
from database import save_processed_data, get_processed_data, bytes_to_dataframe, dataframe_to_bytes
import pandas as pd

app = FastAPI(title="BigData Analytics Backend")

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директории
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class PredictionRequest(BaseModel):
    country_code: str
    target_series: str = "GDP (current US$)"
    exclude_series: list = ["GNI, Atlas method (current US$)"]
    years_count: int = 5


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Сохранение файла
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Предобработка
    try:
        processed_df = preprocess_data_spark(file_path)

        # Извлечение кода страны
        country_code = processed_df['Country Code'].iloc[0] if 'Country Code' in processed_df.columns else "UNK"

        # Кэширование
        data_bytes = dataframe_to_bytes(processed_df)
        save_processed_data(file.filename, country_code, data_bytes)

        return {
            "file_id": file.filename,
            "country_code": country_code,
            "message": "File processed and cached successfully"
        }
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.post("/predict")
async def predict_data(request: PredictionRequest):
    # Получение данных из кэша
    data_bytes = get_processed_data(request.country_code + ".csv")
    if not data_bytes:
        raise HTTPException(404, "Data not found in cache")

    df = bytes_to_dataframe(data_bytes)

    # Фильтрация по стране
    country_df = df[df['Country Code'] == request.country_code]
    if country_df.empty:
        raise HTTPException(404, f"Data for country {request.country_code} not found")

    # Подготовка данных для модели
    try:
        X_all, y_all, X_future = prepare_features_target(
            df=country_df,
            target_series_name=request.target_series,
            exclude_series=request.exclude_series,
            prediction_years_count=request.years_count
        )

        # Обучение и прогнозирование
        results = train_and_predict_model(X_all, y_all, X_future)

        # Форматирование ответа
        plot_data = results['plot_data']
        return {
            "historical": plot_data[['Year', 'Historical']].dropna().to_dict(orient='records'),
            "forecast": plot_data[['Year', 'Median', 'Lower', 'Upper']].dropna().to_dict(orient='records'),
            "country": country_df['Country Name'].iloc[0],
            "target": request.target_series
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@app.get("/countries")
async def list_countries():
    # Получение списка обработанных стран
    # В реальном проекте нужно реализовать сканирование кэша
    return {"countries": ["RUS", "USA", "CHN"]}  # Заглушка


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)