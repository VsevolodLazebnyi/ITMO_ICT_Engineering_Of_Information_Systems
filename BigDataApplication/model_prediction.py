import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objs as go
from data_preprocessing import preprocess_data


def prepare_features_target(df: pd.DataFrame, target_series_name: str, exclude_series: list,
                            prediction_years_count: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Подготавливает данные для обучения и прогнозирования.

    Args:
        df (pd.DataFrame): входной датафрейм после предобработки.
        target_series_name (str): название целевого ряда.
        exclude_series (list): список рядов, которые нужно исключить из признаков.
        prediction_years_count (int): количество лет для прогнозирования.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            - X_all (pd.DataFrame): исторические признаки.
            - y_all (pd.Series): историческая целевая переменная.
            - X_future (pd.DataFrame): признаки для будущих годов (для прогнозирования).
    """

    # Определение исторических годов
    # Извлекаем года как строки, потому что они являются строковыми именами колонок
    historical_year_strings = sorted([col for col in df.columns if str(col).isdigit()])
    # Преобразуем в int для логики определения последнего года и создания будущих годов
    historical_years_int = [int(y) for y in historical_year_strings]

    # Последний исторический год
    last_historical_year = max(
        historical_years_int) if historical_years_int else 2023  # Запасной вариант, если нет годов

    # Создание списка будущих годов
    future_years_list_int = list(range(last_historical_year + 1, last_historical_year + 1 + prediction_years_count))

    # Поиск целевого ряда
    gdp_series_row = df[df['Series Name'] == target_series_name]

    if gdp_series_row.empty:
        raise ValueError(f"Целевой ряд '{target_series_name}' не найден в данных после предобработки.")

    # Выбираем колонки, используя их строковые имена
    y_all = gdp_series_row[historical_year_strings].iloc[0].T.rename(target_series_name)
    y_all.index = y_all.index.astype(int)  # Индекс y_all должен быть int для последующих операций

    # X_all - исторические признаки
    feature_series_names = [
        name for name in df['Series Name']
        if name != target_series_name and not any(ex in name for ex in exclude_series)
    ]

    if not feature_series_names:
        print("Не найдено подходящих признаков для модели, кроме целевого ряда. X_all будет пустым.")
        X_all = pd.DataFrame(index=y_all.index)
    else:
        # Выбираем признаки из датафрейма
        X_all = df[df['Series Name'].isin(feature_series_names)][historical_year_strings].T
        X_all.columns = df[df['Series Name'].isin(feature_series_names)]['Series Name'].tolist()
        X_all.index = X_all.index.astype(int)  # Индекс X_all должен быть int для последующих операций

    # X_future должен иметь те же колонки, что и X_all.
    # Индекс X_future - будущие годы (int).
    X_future = pd.DataFrame(0, index=future_years_list_int, columns=X_all.columns)

    return X_all, y_all, X_future


def train_and_predict_model(X_all: pd.DataFrame, y_all: pd.Series, X_future: pd.DataFrame,
                            bootstrap_iterations: int = 150) -> dict:
    """
    Обучает модель GradientBoostingRegressor и делает предсказания с доверительными интервалами.

    Args:
        X_all (pd.DataFrame): исторические признаки.
        y_all (pd.Series): историческая целевая переменная.
        X_future (pd.DataFrame): признаки для будущих годов.
        bootstrap_iterations (int): количество итераций для бутстраппинга.

    Returns:
        dict: словарь с датафрейм для построения графика и числовыми предсказаниями.
    """

    # Объединяем исторические и будущие признаки для масштабирования
    if X_all.empty and not X_future.empty:
        # Если X_all пуст, но X_future не пуст (т.е. есть прогнозные годы, но нет исторических признаков)
        print("X_all пуст. Модель GradientBoostingRegressor не может быть обучена без признаков.")
        # Возвращаем пустые результаты, чтобы избежать дальнейших ошибок
        return {
            'plot_data': pd.DataFrame(),
            'forecast_values': pd.DataFrame()
        }
    elif X_all.empty and X_future.empty:
        print("X_all и X_future пусты. Нет данных для обучения или прогноза.")
        return {
            'plot_data': pd.DataFrame(),
            'forecast_values': pd.DataFrame()
        }

    # Если X_all не пуст, но есть проблема с размером (например, нет колонок)
    if X_all.shape[1] == 0:
        print("X_all не содержит признаков. Модель GradientBoostingRegressor требует признаков.")
        return {
            'plot_data': pd.DataFrame(),
            'forecast_values': pd.DataFrame()
        }

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_future_scaled = scaler.transform(X_future)

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    bootstrapped_predictions = []

    # Бутстраппинг для оценки интервалов
    for _ in range(bootstrap_iterations):
        sample_idx = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
        X_train = X_scaled[sample_idx]
        y_train = y_all.iloc[sample_idx]

        model.fit(X_train, y_train)

        # Предсказываем на всем объеме (исторические + будущие)
        pred = model.predict(np.vstack([X_scaled, X_future_scaled]))
        bootstrapped_predictions.append(pred)

    predictions = np.array(bootstrapped_predictions)
    lower = np.quantile(predictions, 0.05, axis=0)
    upper = np.quantile(predictions, 0.95, axis=0)
    median = np.quantile(predictions, 0.5, axis=0)

    # Корректировка медианного прогноза с учетом роста
    future_years_index = X_future.index.values  # Годы прогноза
    growth_factor = 1.05  # 5% ежегодного роста

    adjusted_median = median.copy()

    # Применяем множитель только к прогнозной части
    start_idx_forecast = len(y_all)

    if len(future_years_index) > 0:
        base_future_year = future_years_index[0]  # Первый год прогноза
        for i, year in enumerate(future_years_index):
            idx_in_pred = start_idx_forecast + i  # Индекс прогнозного значения в массиве pred
            multiplier = np.exp((year - base_future_year) * np.log(growth_factor))
            adjusted_median[idx_in_pred] *= multiplier

    # Создание датафрейм для Plotly
    all_years_combined = pd.Index(
        list(X_all.index) + list(X_future.index)).unique().sort_values()
    plot_df = pd.DataFrame({
        'Year': all_years_combined,
        'Historical': y_all.reindex(all_years_combined, fill_value=np.nan),
        'Median': pd.Series(adjusted_median, index=all_years_combined),
        'Lower': pd.Series(lower, index=all_years_combined),
        'Upper': pd.Series(upper, index=all_years_combined)
    })

    # Извлечение только прогнозных числовых значений
    forecast_df = plot_df[plot_df['Year'].isin(X_future.index)].copy()

    return {
        'plot_data': plot_df,
        'forecast_values': forecast_df[['Year', 'Median', 'Lower', 'Upper']]
    }


def create_forecast_plot(plot_df: pd.DataFrame, target_series_name: str, plot_title: str) -> go.Figure:
    """
    Создает интерактивный график прогноза с помощью Plotly.

    Args:
        plot_df (pd.DataFrame): датафрейм, содержащий данные для построения графика.
        target_series_name (str): название целевого показателя для подписи оси Y.
        plot_title (str): заголовок графика.

    Returns:
        go.Figure: объект Plotly Figure.
    """
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


def create_all_series_normalized_plot(df: pd.DataFrame) -> go.Figure:
    """
    Создает нормализованный график всех временных рядов.

    Args:
        df (pd.DataFrame): предобработанный датафрейм с данными.

    Returns:
        go.Figure: объект Plotly Figure.
    """
    # Определяем колонки, которые являются годами
    year_columns = [col for col in df.columns if str(col).isdigit()]

    df_for_melt = df[['Series Name'] + year_columns].copy()

    long_df = df_for_melt.melt(id_vars=["Series Name"], var_name="Year", value_name="Value")

    long_df["Year"] = long_df["Year"].astype(int)

    # Убедимся, что нормализация обрабатывает NaN
    def minmax_norm(s):
        # Пропускаем NaN при расчете min/max
        if s.dropna().empty:
            return s
        return (s - s.min()) / (s.max() - s.min())

    # Применяем нормализацию, игнорируя NaN
    long_df["Normalized"] = long_df.groupby("Series Name")["Value"].transform(minmax_norm)

    # Удаляем ряды, которые полностью состоят из NaN после нормализации (т.е. были NaN изначально)
    # или не имели достаточно вариаций для нормализации (s.max() - s.min() было 0)
    long_df.dropna(subset=["Normalized"], inplace=True)

    fig = px.line(
        long_df,
        x="Year",
        y="Normalized",
        color="Series Name",
        title="Min-Max нормализованные траектории всех показателей",
        labels={"Normalized": "Нормализованное значение", "Year": "Год"}
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            title="Показатели",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        xaxis=dict(showgrid=True, dtick=2),
        yaxis=dict(showgrid=True),
        height=800,
        margin=dict(r=220)
    )

    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Год: %{x}<br>Нормал. значение: %{y:.2f}'
    )
    return fig


# Пример использования для тестирования (можно удалить)
if __name__ == "__main__":
    # input_file_path = "data/f4dc79c5-2efc-4406-957a-e8dc7fc523fc_Series - Metadata.csv"
    input_file_path = "data/2a4b8717-1da4-453d-ba54-0e3edf91215c_Series - Metadata.csv"
    data_cleaned = preprocess_data(input_file_path)

    # Обработка отсутствующего целевого ряда
    target_name = 'GDP (current US$)'
    available_series = data_cleaned['Series Name'].unique()

    if target_name not in available_series:
        print(f"\nЦелевой ряд '{target_name}' не найден в предобработанных данных.")
        print("Доступные ряды (фрагмент):")
        # Выводим первые 10 уникальных названий серий для удобства
        for i, series in enumerate(available_series[:10]):
            print(f"- {series}")
        if len(available_series) > 10:
            print(f"... и еще {len(available_series) - 10} рядов.")

        # Предлагаем использовать 'Population, total' в качестве целевого ряда для теста
        if 'Population, total' in available_series:
            print(f"\nВ качестве целевого ряда для теста будет использоваться 'Population, total'.")
            target_name = 'Population, total'
        else:
            print("\nВыберите один из доступных рядов выше и измените 'target_name' в коде.")
            exit()  # Останавливаем выполнение

    excluded_series = ['GNI, Atlas method (current US$)']

    # Подготовка данных для модели
    X_all_data, y_all_data, X_future_data = prepare_features_target(
        df=data_cleaned,
        target_series_name=target_name,
        exclude_series=excluded_series
    )

    print("\nДанные для модели")
    print("X_all_data head:\n", X_all_data.head())
    print("\ny_all_data head:\n", y_all_data.head())
    print("\nX_future_data head:\n", X_future_data.head())

    # Обучение модели и получение предсказаний
    model_results = train_and_predict_model(
        X_all=X_all_data,
        y_all=y_all_data,
        X_future=X_future_data,
        bootstrap_iterations=150
    )

    plot_df_result = model_results['plot_data']
    forecast_values_df = model_results['forecast_values']

    if not plot_df_result.empty:
        print("\nПрогнозные значения:")
        print(forecast_values_df)

        # Визуализация прогноза
        forecast_fig = create_forecast_plot(plot_df_result, target_name, f'Прогноз {target_name} на 5 лет')
        # HTML-представление:
        pio.write_html(forecast_fig, file='gdp_forecast.html', auto_open=True)

        # Визуализация всех нормализованных показателей
        all_series_fig = create_all_series_normalized_plot(data_cleaned)
        pio.write_html(all_series_fig, file='all_series_normalized.html', auto_open=True)
    else:
        print("\nВизуализация не создана, так как модель не была обучена.")