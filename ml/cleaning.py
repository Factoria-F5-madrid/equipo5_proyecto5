import pandas as pd
import numpy as np
from pathlib import Path

file = Path("data/Life Expectancy Data.csv")

if file.exists():

else:

    exit(1)

df = pd.read_csv(file)

df = df.drop_duplicates()

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

output_file = "data/clean_data.csv"
df.to_csv(output_file, index=False)

