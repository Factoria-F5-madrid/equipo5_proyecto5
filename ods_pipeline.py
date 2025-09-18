import pandas as pd
import numpy as np
import json

WEIGHTS = {"health": 0.4, "education": 0.3, "economy": 0.3}

def minmax(series, invert=False):
    if invert:
        series = series.max() - series
    return 100 * (series - series.min()) / (series.max() - series.min())

def process_ods(df: pd.DataFrame) -> pd.DataFrame:
    # Imputar faltantes
    for col in ["Life expectancy ", "GDP", "Schooling", "Adult Mortality", "infant deaths", "Income composition of resources"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Variables derivadas
    df['log_gdp'] = np.log1p(df['GDP'])
    df['health_score'] = (
        minmax(df['Life expectancy ']) +
        minmax(df['Adult Mortality'], invert=True) +
        minmax(df['infant deaths'], invert=True)
    ) / 3
    df['education_score'] = minmax(df['Schooling'])
    df['economy_score'] = (minmax(df['log_gdp']) + minmax(df['Income composition of resources'])) / 2

    # Índice ODS
    df['ods_index'] = (
        WEIGHTS['health']*df['health_score'] +
        WEIGHTS['education']*df['education_score'] +
        WEIGHTS['economy']*df['economy_score']
    ).clip(0,100)

    # Guardar metadata
    metadata = {
        "weights": WEIGHTS,
        "rows_processed": len(df),
        "columns_added": ["log_gdp","health_score","education_score","economy_score","ods_index"]
    }
    with open("data/ods_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return df