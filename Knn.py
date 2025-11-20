from numpy import sort
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    ConfusionMatrixDisplay
)

## Configurations
train_filename = "./data/cleveland_train.csv"
test_filename = "./data/cleveland_test.csv"
targets = ["target"]
n_neighbors = 9

train_dataset = pd.read_csv(train_filename)
test_dataset = pd.read_csv(test_filename)

x_train = train_dataset.drop(columns=targets)
t_train = train_dataset[targets] # real

x_test = test_dataset.drop(columns=targets)
t_test = test_dataset[targets] # real

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(x_train, t_train)

y_train = knn.predict(x_train) # model output
y_test = knn.predict(x_test) # model output

classes = train_dataset['target'].unique()


def evaluate_model(real, predicted, dataset_name):
    print(f"\n--- Métricas para {dataset_name} ---")
    print(f"Acurácia:          {accuracy_score(real, predicted):.3f}")
    print(f"F1-score (macro):  {f1_score(real, predicted, average='macro'):.3f}")
    print(f"F1-score (weighted): {f1_score(real, predicted, average='weighted'):.3f}")
    print(f"Recall (macro):    {recall_score(real, predicted, average='macro'):.3f}")
    print(f"Precision (macro): {precision_score(real, predicted, average='macro'):.3f}")

    # Matriz de confusão
    cm = ConfusionMatrixDisplay.from_predictions(real, predicted, labels=classes)
    cm.ax_.set_title(f"Matriz de Confusão: {dataset_name}")
    plt.show()

# Avaliar treino e teste
evaluate_model(t_train, y_train, "Treino")
evaluate_model(t_test, y_test, "Teste")

def display_confusion_matrix(real, model_output, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(real, model_output, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()

display_confusion_matrix(t_train, y_train, classes, "Train confusion matrix")
display_confusion_matrix(t_test, y_test, classes, "Test confusion matrix")
