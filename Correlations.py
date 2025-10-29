import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Configurations
dataset_filename = "./data/cleveland.csv"
num_outputs = 1

dataset = pd.read_csv(dataset_filename)
columns = dataset.columns
num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

# One-Hot Encoding
classes = dataset["target"].unique()
dataset = pd.get_dummies(dataset, columns=["target"])

corr_matrix = dataset.corr() * 100
print(corr_matrix)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Correlation Heatmap')
plt.show()