from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px
import io
import joblib
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Spark
spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()


def clean_year_col(col_name):
    if '[' in col_name and 'YR' in col_name:
        return col_name.split(' ')[0].strip()
    return col_name


def fill_na_neighbors(df_subset, degree=2):
    df_filled = df_subset.copy()
    years = np.array([int(col) for col in df_filled.columns])
    dtypes = {col: df_filled[col].dtype for col in df_filled.columns}

    for idx, row in df_filled.iterrows():
        vals = row.values
        mask = ~pd.isna(vals)

        if mask.sum() == 0:
            continue
        is_integer = np.issubdtype(row.dtype, np.integer)
        X_known = years[mask].reshape(-1, 1)
        y_known = vals[mask].astype(float)

        if mask.sum() > degree + 1:
            model = make_pipeline(
                PolynomialFeatures(degree),
                LinearRegression()
            )
        else:
            model = LinearRegression()

        model.fit(X_known, y_known)
        preds = model.predict(years.reshape(-1, 1))

        if not is_integer:
            new_vals = np.where(~mask, preds, vals)
        else:
            new_vals = np.where(~mask, np.round(preds).astype(int), vals)
            new_vals = np.clip(new_vals, a_min=0, a_max=None)

        for i, col in enumerate(df_filled.columns):
            df_filled.at[idx, col] = dtypes[col].type(new_vals[i])

    return df_filled


def preprocess_data_spark(file_path):
    logger.info(f"Начало обработки файла: {file_path}")

    # Чтение данных с PySpark
    df = spark.read.csv(file_path, header=True, inferSchema=True, encoding='latin1')

    # Удаление дубликатов
    df = df.dropDuplicates()

    # Очистка строковых колонок
    str_cols = [f.name for f in df.schema if f.dataType == StringType()]
    for col in str_cols:
        df = df.withColumn(col, F.trim(F.col(col)))

    # Удаление колонок с >50% пропусков
    total_rows = df.count()
    cols_to_drop = []
    for col in df.columns:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count / total_rows > 0.5:
            cols_to_drop.append(col)
    df = df.drop(*cols_to_drop)

    # Заполнение пропусков в строковых колонках
    for col in str_cols:
        df = df.withColumn(col, F.when(F.col(col).isNull(), '').otherwise(F.col(col)))

    # Заполнение пропусков в числовых колонках медианой
    numeric_cols = [f.name for f in df.schema if f.dataType != StringType()]
    for col in numeric_cols:
        median_value = df.approxQuantile(col, [0.5], 0.01)[0]
        df = df.withColumn(col, F.when(F.col(col).isNull(), median_value).otherwise(F.col(col)))

    # Очистка названий колонок (годы)
    new_columns = [clean_year_col(col) for col in df.columns]
    df = df.toDF(*new_columns)

    # Обработка годовых колонок
    year_cols = [col for col in df.columns if col.isdigit()]
    invalid_values = ['..', '-', 'N/A', 'NA', 'n/a', 'na', '']

    # Приведение типов и замена недопустимых значений
    for col in year_cols:
        df = df.withColumn(
            col,
            F.when(F.col(col).isin(invalid_values), None)
            .otherwise(F.col(col))
            .cast(DoubleType())
        )

    # Конвертация в Pandas для заполнения пропусков
    pdf = df.toPandas()

    # Заполнение пропусков в годовых данных
    if year_cols:
        pdf[year_cols] = fill_na_neighbors(pdf[year_cols])

    # Удаление строк, где все годовые данные отсутствуют
    pdf = pdf.dropna(subset=year_cols, how='all')

    # Удаление строк, где строковые колонки пусты
    str_cols_pdf = pdf.select_dtypes(include='object').columns
    pdf = pdf[pdf[str_cols_pdf].ne('').any(axis=1)]

    logger.info(f"Обработка завершена. Размер данных: {pdf.shape}")
    return pdf


def prepare_features_target(df: pd.DataFrame, target_series_name: str, exclude_series: list,
                            prediction_years_count: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    historical_year_strings = sorted([col for col in df.columns if str(col).isdigit()])
    historical_years_int = [int(y) for y in historical_year_strings]
    last_historical_year = max(historical_years_int) if historical_years_int else 2023

    future_years_list_int = list(range(last_historical_year + 1, last_historical_year + 1 + prediction_years_count))

    gdp_series_row = df[df['Series Name'] == target_series_name]
    if gdp_series_row.empty:
        raise ValueError(f"Целевой ряд '{target_series_name}' не найден в данных после предобработки.")

    y_all = gdp_series_row[historical_year_strings].iloc[0].T.rename(target_series_name)
    y_all.index = y_all.index.astype(int)

    feature_series_names = [
        name for name in df['Series Name']
        if name != target_series_name and not any(ex in name for ex in exclude_series)
    ]

    if not feature_series_names:
        print("Не найдено подходящих признаков для модели, кроме целевого ряда. X_all будет пустым.")
        X_all = pd.DataFrame(index=y_all.index)
    else:
        X_all = df[df['Series Name'].isin(feature_series_names)][historical_year_strings].T
        X_all.columns = df[df['Series Name'].isin(feature_series_names)]['Series Name'].tolist()
        X_all.index = X_all.index.astype(int)

    X_future = pd.DataFrame(0, index=future_years_list_int, columns=X_all.columns)

    return X_all, y_all, X_future


def train_and_predict_model(X_all: pd.DataFrame, y_all: pd.Series, X_future: pd.DataFrame,
                            bootstrap_iterations: int = 150) -> dict:
    if X_all.empty and not X_future.empty:
        return {'plot_data': pd.DataFrame(), 'forecast_values': pd.DataFrame()}
    elif X_all.empty and X_future.empty:
        return {'plot_data': pd.DataFrame(), 'forecast_values': pd.DataFrame()}

    if X_all.shape[1] == 0:
        return {'plot_data': pd.DataFrame(), 'forecast_values': pd.DataFrame()}

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_future_scaled = scaler.transform(X_future)

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    bootstrapped_predictions = []

    for _ in range(bootstrap_iterations):
        sample_idx = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
        X_train = X_scaled[sample_idx]
        y_train = y_all.iloc[sample_idx]

        model.fit(X_train, y_train)
        pred = model.predict(np.vstack([X_scaled, X_future_scaled]))
        bootstrapped_predictions.append(pred)

    predictions = np.array(bootstrapped_predictions)
    lower = np.quantile(predictions, 0.05, axis=0)
    upper = np.quantile(predictions, 0.95, axis=0)
    median = np.quantile(predictions, 0.5, axis=0)

    future_years_index = X_future.index.values
    growth_factor = 1.05
    adjusted_median = median.copy()
    start_idx_forecast = len(y_all)

    if len(future_years_index) > 0:
        base_future_year = future_years_index[0]
        for i, year in enumerate(future_years_index):
            idx_in_pred = start_idx_forecast + i
            multiplier = np.exp((year - base_future_year) * np.log(growth_factor))
            adjusted_median[idx_in_pred] *= multiplier

    all_years_combined = pd.Index(list(X_all.index) + list(X_future.index)).unique().sort_values()
    plot_df = pd.DataFrame({
        'Year': all_years_combined,
        'Historical': y_all.reindex(all_years_combined, fill_value=np.nan),
        'Median': pd.Series(adjusted_median, index=all_years_combined),
        'Lower': pd.Series(lower, index=all_years_combined),
        'Upper': pd.Series(upper, index=all_years_combined)
    })

    forecast_df = plot_df[plot_df['Year'].isin(X_future.index)].copy()

    return {
        'plot_data': plot_df,
        'forecast_values': forecast_df[['Year', 'Median', 'Lower', 'Upper']]
    }


def create_forecast_plot(plot_df: pd.DataFrame, target_series_name: str, plot_title: str) -> go.Figure:
    fig = px.line(plot_df, x='Year', y='Historical',
                  title=plot_title,
                  labels={'value': target_series_name, 'Year': 'Год'})

    fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Median'], mode='lines',
                             name='Медианный прогноз', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Upper'], mode='lines',
                             name='Верхний интервал', line=dict(width=0),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Lower'], mode='lines',
                             name='Нижний интервал', fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                             line=dict(width=0), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        hovermode="x unified",
        showlegend=True,
        height=600,
        xaxis_title="Год",
        yaxis_title=target_series_name,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.5)', borderwidth=1)
    )

    return fig


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(df, buffer)
    return buffer.getvalue()


def bytes_to_dataframe(data_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(data_bytes)
    return joblib.load(buffer)