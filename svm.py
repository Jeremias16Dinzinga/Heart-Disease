import pandas as pd
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

## Configurations
train_filename = "./data/cleveland_train.csv"
test_filename = "./data/cleveland_test.csv"
targets = ["target"]
C = 1
kernel = 'rbf'

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets].squeeze()

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets].squeeze()

model = svm.SVC(C=C, kernel=kernel)
model.fit(x_train, t_train)

y_train = model.predict(x_train)
y_test = model.predict(x_test)

classes = sorted(train_dataset['target'].unique())

# ----- MATRIZ DE CONFUSÃO -----

def display_confusion_matrix(real, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        real,
        predicted,
        labels=classes
    )
    cm.ax_.set_title(title)
    plt.show()

# Treino
display_confusion_matrix(t_train, y_train, classes, "Train - Matriz de Confusão")

# Teste
display_confusion_matrix(t_test, y_test, classes, "Test - Matriz de Confusão")