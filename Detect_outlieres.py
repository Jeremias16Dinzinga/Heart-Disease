import pandas as pd

dataset_filename = "./data/cleveland.csv"
dataset = pd.read_csv(dataset_filename)

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"{column}: {len(outliers)} outliers")
    return outliers

# Exemplo com as principais variáveis
for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    detect_outliers_iqr(dataset, col)
