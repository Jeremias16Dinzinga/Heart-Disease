import pandas as pd

dataset_filename = "./data/cleveland.csv"
dataset = pd.read_csv(dataset_filename)

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Aplicar para variáveis com outliers
for col in column_names:
    dataset = remove_outliers_iqr(dataset, col)

dataset.to_csv(
    "./data/cleveland.csv",
    index=False,
)
print(f"Novo número de linhas: {len(dataset)}")
