#!/usr/bin/env python3
"""
Script to execute data cleaning from Person 1's notebook
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== STARTING DATA CLEANING ===")

# Load the csv
file = Path("data/Life Expectancy Data.csv")

if file.exists():
    print(f"Loading file: {file}")
else:
    print(f"Error: File not found {file}")
    exit(1)

df = pd.read_csv(file)

# Quick overview
print(f"(rows, columns): {df.shape}")

# Check duplicates
print(f"Duplicates found: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Check missing values
print("\nMissing values per column:")
print(df.isna().sum())

# Impute missing values
print("\nImputing missing values...")

# numerical → mean
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# categorical → most frequent value (mode)
cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after imputation:")
print(df.isna().sum().sum())

# Normalize column names
print("\nNormalizing column names...")
df.columns = (
    df.columns.str.strip()        # remove leading/trailing spaces
    .str.lower()                  # lowercase
    .str.replace(" ", "_")        # spaces to _
    .str.replace("-", "_")        # hyphens to _
)

print("Normalized column names:")
print(df.columns.tolist())

# Save clean dataset
output_file = "data/clean_data.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Clean file saved as: {output_file}")
print(f"Final shape: {df.shape}")

print("\n=== DATA CLEANING COMPLETED ===")
