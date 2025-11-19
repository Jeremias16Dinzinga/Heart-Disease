from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import pandas as pd

## Configurations
dataset_filename = "./data/cleveland.csv"
train_dataset = "./data/cleveland_train.csv"
test_dataset = "./data/cleveland_test.csv"
target = "target"

# Load datasets
train_dataset = pd.read_csv(train_dataset)
test_dataset = pd.read_csv(test_dataset)

# Split input and target
x_train = train_dataset.drop(columns=[target])
t_train = train_dataset[target]

# Create pipeline (NORMALIZATION + MODEL)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Parameters for GridSearch
param = {
    'knn__n_neighbors': range(1, 21),
    'knn__p': [1, 2, 3, 4, 5]
}

# GridSearchCV
gs = GridSearchCV(
    pipeline,
    param,
    scoring='f1_macro',
    verbose=True,
    cv=5
)

# Fit
gs.fit(x_train, t_train)

# Results
print("Best parameters found:", gs.best_params_)
print("Best score (F1-macro):", gs.best_score_)
