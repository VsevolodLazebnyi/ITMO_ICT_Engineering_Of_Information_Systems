import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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


def clean_year_col(col_name):
    if '[' in col_name and 'YR' in col_name:
        return col_name.split(' ')[0].strip()
    return col_name


def preprocess_data(input_file_path):
    """
    Загружает данные, выполняет предобработку и возвращает очищенный DataFrame.
    """
    df = pd.read_csv(input_file_path, encoding='latin1', on_bad_lines='warn')

    df = df.drop_duplicates()

    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    df = df.loc[:, df.isnull().mean() < 0.5]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())

    df.columns = [clean_year_col(col) for col in df.columns]

    year_cols = [col for col in df.columns if col.isdigit()]

    invalid_values = ['..', '-', 'N/A', 'NA', 'n/a', 'na', '']
    for col in year_cols:
        df[col] = df[col].replace(invalid_values, np.nan)

    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')

    if year_cols:
        df_years_filled = fill_na_neighbors(df[year_cols])
        df[year_cols] = df_years_filled

    df = df.dropna()

    str_cols = df.select_dtypes(include='object').columns
    df = df.loc[~(df[str_cols] == '').all(axis=1)]

    return df

# Пример использования для тестирования (можно удалить)
if __name__ == "__main__":
    test_file_path = "data/f4dc79c5-2efc-4406-957a-e8dc7fc523fc_Series - Metadata.csv"


    cleaned_df = preprocess_data(test_file_path)
    print(cleaned_df.head())
    print(cleaned_df.info())
    print(cleaned_df['Series Name'].unique())
    cleaned_df.to_csv("cleaned_metadata_for_test.csv", index=False) # Если нужно сохранить в отдельный файл