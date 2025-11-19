import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

## Configurations
dataset_filename = "./data/cleveland.csv"
train_dataset = "./data/cleveland_train.csv"
test_dataset = "./data/cleveland_test.csv"
targets = ["target"]
perc_train = 50/100

dataset = pd.read_csv(dataset_filename)

x = dataset.drop(columns=targets)
t = dataset[targets]

column_names = x.columns

print(t.value_counts())

x_train, x_test, t_train, t_test = train_test_split(
    x, t,
    train_size = perc_train, stratify = t
)

# Guardar o índice original antes da transformação
x_train_original = x_train.copy()
x_test_original = x_test.copy()

imputer = SimpleImputer(strategy="mean")

# x_train precisa ajustar e transformar
x_train = imputer.fit_transform(x_train)

# x_test só transforma
x_test = imputer.transform(x_test)

# Transformar de volta em DataFrame para manter colunas e índice
x_train = pd.DataFrame(x_train, columns=column_names, index=x_train_original.index)
x_test  = pd.DataFrame(x_test, columns=column_names, index=x_test_original.index)

# Data rescale
scaler = StandardScaler().fit(x_train)

x_train_scaled = pd.DataFrame(
    scaler.transform(x_train),
    columns= column_names,
    index = x_train.index
)

x_test_scaled = pd.DataFrame(
    scaler.transform(x_test),
    columns= column_names,
    index = x_test.index
)


train = pd.concat([x_train, t_train], axis='columns', join='inner')
test = pd.concat([x_test, t_test], axis='columns', join='inner')

train.to_csv(train_dataset, index=False)
test.to_csv(test_dataset, index=False)
