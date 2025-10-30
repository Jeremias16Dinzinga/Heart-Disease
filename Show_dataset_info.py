import pandas as pd

dataset = pd.read_csv("./data/cleveland.data", header=None)

# Check data types and missing data
dataset.info()

# Print statistic information
#print(dataset.describe(include="all"))