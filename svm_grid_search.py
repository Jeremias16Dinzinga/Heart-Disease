import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm

## Configurations
train_filename = "./data/cleveland_train.csv"
targets = ["target"] 

C_values_search = np.logspace(-2, 2, 10)  
# Isto gera valores entre 0.01 e 100 — ideal para Heart Disease

# Grid de hiperparâmetros
param = [
    {
        'C': C_values_search,
        'gamma': ['scale', 0.01, 0.05, 0.1, 0.5],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
]

train_dataset = pd.read_csv(train_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets].squeeze()

gs = GridSearchCV(
    svm.SVC(),
    param,
    scoring='f1_macro',
    cv=5,
    verbose=True,
    n_jobs=-1,
    error_score='raise'
)

# Treinar
gs.fit(x_train, t_train)

# Resultados
print("\nBest parameters found: ", gs.best_params_)
print("Best F1-score: ", gs.best_score_)
