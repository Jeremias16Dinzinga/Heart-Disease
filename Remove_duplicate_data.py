import pandas as pd

# Carregar o dataset Heart Disease
dataset = pd.read_csv("./data/cleveland.csv")

# Contar linhas iniciais
initial_rows = len(dataset)

# Remover duplicados
dataset = dataset.drop_duplicates()

# Contar linhas finais
rows = len(dataset)
duplicated_rows = initial_rows - rows

# Mostrar resultados
if duplicated_rows > 0:
    print(f"Removed {duplicated_rows} duplicated samples")
    dataset.to_csv(
         "./data/cleveland.csv",
        index=False,
    )

print(f"Dataset contains {rows} samples")
