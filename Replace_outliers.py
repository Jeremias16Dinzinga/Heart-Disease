import pandas as pd

dataset_filename_train = "./data/cleveland_train.csv"
dataset_filename_test = "./data/cleveland_test.csv"
dataset_filename = "./data/cleveland.csv"
dataset = pd.read_csv(dataset_filename)

continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)
    return df

for col in continuous_cols:
    dataset = cap_outliers_iqr(dataset, col)

dataset.to_csv(
    "./data/cleveland.csv",
    index=False,
)