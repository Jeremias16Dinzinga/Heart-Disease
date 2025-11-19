import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Carregar dataset original
dataset_filename = "./data/cleveland.csv"

dataset = pd.read_csv(dataset_filename)

# Colunas contínuas (numéricas)
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Definir feature e target
target_col = 'target'
X = dataset.drop(columns=[target_col])
y = dataset[target_col]

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aplicar RobustScaler
scaler = RobustScaler().fit(x_train[continuous_cols])

# Escalar apenas as colunas contínuas
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[continuous_cols] = scaler.transform(x_train[continuous_cols])
x_test_scaled[continuous_cols] = scaler.transform(x_test[continuous_cols])

# Recriar os datasets completos (features + target)
train_scaled = pd.concat([x_train_scaled, y_train], axis=1)
test_scaled = pd.concat([x_test_scaled, y_test], axis=1)

# Guardar os ficheiros
train_scaled.to_csv("./data/cleveland_train_scaled.csv", index=False)
test_scaled.to_csv("./data/cleveland_test_scaled.csv", index=False)

print("Normalização robusta concluída!")
print(f"Tamanho do treino: {x_train.shape[0]} linhas")
print(f"Tamanho do teste:  {x_test.shape[0]} linhas")
print("Ficheiros salvos em:")
print(" - ./data/cleveland_train_scaled.csv")
print(" - ./data/cleveland_test_scaled.csv")
