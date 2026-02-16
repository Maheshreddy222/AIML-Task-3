# Import libraries
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("Housing.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset info
print("\nDataset Info:")
print(df.info())

# Display missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# Step 2: Handle missing values

# Fill numerical columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# Step 3: Convert categorical features into numerical
df = pd.get_dummies(df, drop_first=True)


# Display processed dataset
print("\nPreprocessed Dataset:")
print(df.head())

print("\nFinal Shape:", df.shape)
