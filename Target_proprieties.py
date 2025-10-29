import matplotlib.pyplot as plt
import pandas as pd

## Configurations
dataset_filename = "./data/cleveland.csv"
num_outputs = 1

dataset = pd.read_csv(dataset_filename)
columns = dataset.columns
num_inputs = len(columns) - num_outputs
input_columns = columns[0:num_inputs]

target_counts = dataset['target'].value_counts()
rows = len(dataset)

# Plot a pie chart
target_counts.plot.pie(
    startangle=90,
    autopct=lambda percent: f"{percent:.2f}% ({percent / 100 * rows:.0f})"
)
plt.title('Target Proportionality')
plt.ylabel('') # Hide the y-label
plt.show()