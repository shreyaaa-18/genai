import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload dataset
uploaded = files.upload()

# Load dataset
df = pd.read_csv("telecom_tower_failure.csv")

# Display first five records
print("First five records:")
display(df.head())

# Dataset information
print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary:")
display(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing numerical values
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Feature 1: High Temperature
df["High_Temperature"] = np.where(
    df["Temperature_C"] > 60,
    1,
    0
)

# Feature 2: Low Battery
df["Low_Battery"] = np.where(
    df["Battery_Voltage"] < 45,
    1,
    0
)

# Feature 3: Power-Traffic Ratio
df["Power_Traffic_Ratio"] = (
    df["Power_Consumption_W"] /
    (df["Traffic_Load"] + 0.01)
)

# Feature 4: Tower Age Category
df["Tower_Age_Category"] = pd.cut(
    df["Tower_Age_Years"],
    bins=[0, 3, 7, np.inf],
    labels=["New", "Medium", "Old"]
)

# One-hot encoding
df = pd.get_dummies(
    df,
    columns=["Tower_Age_Category"],
    dtype=int
)

# Remove Tower ID
df = df.drop(columns=["Tower_ID"])

# Separate features and target
X = df.drop(
    columns=["Failure_Within_48Hrs"]
)

y = df["Failure_Within_48Hrs"]

# Display final dataset
print("\nFeature-engineered dataset:")
display(df.head())

print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])

# Target distribution
print("\nTarget distribution:")
print(y.value_counts())

# Save dataset
df.to_csv(
    "telecom_tower_feature_engineered.csv",
    index=False
)

# Download
files.download(
    "telecom_tower_feature_engineered.csv"
)

print("\nFeature engineering completed successfully!")
