import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("./data/cleveland.csv")

columns = dataset.columns
num_outputs = 1
num_inputs = len(columns) - num_outputs

input_columns = columns[0:num_inputs]

for col in input_columns:
    plt.hist(dataset[col])
    plt.title(f"{col} histogram")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
