import pandas as pd
from sklearn.model_selection import train_test_split

## Configurations
dataset_filename = "./data/cleveland.csv"
train_dataset = "./data/cleveland_train.csv"
test_dataset = "./data/cleveland_test.csv"
targets = ["target"]
perc_train = 50/100

dataset = pd.read_csv(dataset_filename)

x = dataset.drop(columns=targets)
t = dataset[targets]

x_train, x_test, t_train, t_test = train_test_split(
    x, t,
    train_size = perc_train
)

train = pd.concat([x_train, t_train], axis='columns', join='inner')
test = pd.concat([x_test, t_test], axis='columns', join='inner')

train.to_csv(train_dataset, index=False)
test.to_csv(test_dataset, index=False)
