import pandas as pd

dataset = pd.read_csv("./data/cleveland.data", encoding='latin1')

# Check data types and missing data
dataset.info()

# Print statistic information
print(dataset.describe(include="all"))