import pandas as pd
import numpy as np
import logging
import joblib
import io
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_year_col(col_name):
    if '[' in col_name and 'YR' in col_name:
        return col_name.split(' ')[0].strip()
    return col_name


def fill_na_neighbors_series(row, degree=2):
    years = np.array([int(col) for col in row.index])
    vals = row.values
    mask = ~pd.isna(vals)

    if mask.sum() == 0:
        return row

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

    if is_integer:
        new_vals = np.where(~mask, np.round(preds).astype(int), vals)
        new_vals = np.clip(new_vals, 0, None)
    else:
        new_vals = np.where(~mask, preds, vals)

    return pd.Series(new_vals, index=row.index)


def preprocess_data(file_path):
    try:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='warn')

        # Базовая очистка
        df = df.drop_duplicates()

        # Очистка строковых колонок
        str_cols = df.select_dtypes(include='object').columns
        for col in str_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Удаление колонок с >50% пропусков
        total_count = len(df)
        null_percent = df.isnull().mean()
        cols_to_drop = null_percent[null_percent > 0.5].index.tolist()
        df = df.drop(columns=cols_to_drop)

        # Очистка названий колонок
        df.columns = [clean_year_col(col) for col in df.columns]

        # Обработка числовых колонок
        year_cols = [col for col in df.columns if col.isdigit()]
        invalid_values = ['..', '-', 'N/A', 'NA', 'n/a', 'na', '']

        for col in year_cols:
            df[col] = df[col].replace(invalid_values, np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Заполнение пропусков
        if year_cols:
            logger.info("Filling missing values")
            df[year_cols] = df[year_cols].apply(
                lambda row: fill_na_neighbors_series(row),
                axis=1
            )

        # Фильтрация строк
        str_cols = df.select_dtypes(include='object').columns
        df = df.dropna(subset=str_cols, how='all')
        df = df.dropna()

        logger.info(f"Processed DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


def prepare_features_target(df, target_series_name, exclude_series, prediction_years_count=5):
    if 'Series Name' not in df.columns:
        raise ValueError("Column 'Series Name' not found in DataFrame")

    historical_year_strings = sorted([col for col in df.columns if str(col).isdigit()])
    if not historical_year_strings:
        raise ValueError("No year columns found in DataFrame")

    historical_years_int = [int(y) for y in historical_year_strings]
    last_historical_year = max(historical_years_int)
    future_years_list_int = list(range(last_historical_year + 1, last_historical_year + 1 + prediction_years_count))

    gdp_series_row = df[df['Series Name'] == target_series_name]
    if gdp_series_row.empty:
        available_series = df['Series Name'].unique()[:10]
        raise ValueError(
            f"Target series '{target_series_name}' not found. Available: {', '.join(available_series)}"
        )

    y_all = gdp_series_row[historical_year_strings].iloc[0].T
    y_all = y_all.rename(target_series_name)
    y_all.index = y_all.index.astype(int)

    feature_series_names = [
        name for name in df['Series Name'].unique()
        if name != target_series_name and not any(ex in name for ex in exclude_series)
    ]

    if not feature_series_names:
        X_all = pd.DataFrame(index=y_all.index)
    else:
        feature_rows = df[df['Series Name'].isin(feature_series_names)]
        if feature_rows.empty:
            X_all = pd.DataFrame(index=y_all.index)
        else:
            X_all = feature_rows[historical_year_strings].T
            X_all.columns = feature_rows['Series Name'].values
            X_all.index = X_all.index.astype(int)

    X_future = pd.DataFrame(0, index=future_years_list_int, columns=X_all.columns)
    return X_all, y_all, X_future


def train_and_predict_model(X_all, y_all, X_future, bootstrap_iterations=150):
    if X_all.empty or X_future.empty or len(X_all) == 0:
        return {
            'plot_data': pd.DataFrame(),
            'forecast_values': pd.DataFrame()
        }

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

    # Корректировка прогноза
    growth_factor = 1.05
    adjusted_median = median.copy()
    start_idx_forecast = len(y_all)
    future_years_index = X_future.index.values

    if len(future_years_index) > 0:
        base_future_year = future_years_index[0]
        for i, year in enumerate(future_years_index):
            idx_in_pred = start_idx_forecast + i
            multiplier = np.exp((year - base_future_year) * np.log(growth_factor))
            adjusted_median[idx_in_pred] *= multiplier

    all_years_combined = list(X_all.index) + list(X_future.index)
    all_years_combined = sorted(set(all_years_combined))

    plot_df = pd.DataFrame({
        'Year': all_years_combined,
        'Historical': y_all.reindex(all_years_combined),
        'Median': pd.Series(adjusted_median, index=all_years_combined),
        'Lower': pd.Series(lower, index=all_years_combined),
        'Upper': pd.Series(upper, index=all_years_combined)
    })

    forecast_df = plot_df[plot_df['Year'].isin(X_future.index)].copy()
    return {
        'plot_data': plot_df,
        'forecast_values': forecast_df[['Year', 'Median', 'Lower', 'Upper']]
    }


# Сериализация данных
def dataframe_to_bytes(df):
    buffer = io.BytesIO()
    joblib.dump(df, buffer)
    return buffer.getvalue()


def bytes_to_dataframe(data_bytes):
    buffer = io.BytesIO(data_bytes)
    return joblib.load(buffer)