# 🧠 RELATÓRIO DE PRÉ-PROCESSAMENTO – DATASET HEART DISEASE

**Disciplina:** Aprendizagem Automática I  
**Trabalho Escolar**

**Nome:** Jeremias Dinzinga  
**Nº:** 1710393  
**Curso:** Ciência de Dados e Inteligência Artificial  
**Ano:** 2º

---

## 📋 Tarefas Executadas

- Adicionei o ficheiro **“processed.cleveland.data”** do dataset *Cleveland* (renomeado para **Cleveland.data**).  
- Salvei os dados num ficheiro CSV (**Cleveland.csv**) com **14 colunas**.  
- Verifiquei que **não há dados duplicados**.  
- Analisei as informações do dataset (**14 colunas e 302 registos válidos**).  
- Criei **histogramas** para cada variável e melhorei a exibição para mostrar **4 por vez**.  
- Adicionei uma **função** para controlar quantas variáveis são exibidas por vez nos histogramas.  
- Mostrei a **distribuição das classes** usando a variável `target` (presença ou ausência de doença cardíaca).  
- Criei uma função para **detectar outliers** com base no **1º e 3º quartil (IQR)**.  
- Visualizei a **proporção das classes (targets)**.  
- Analisei a **correlação entre as variáveis** e também a correlação individual de cada variável com o `target`.  
- Criei uma função para **remover outliers**, caso seja necessário futuramente.  
- Como muitos valores foram identificados como *outliers* (mas clinicamente plausíveis), **optei por não removê-los** para evitar perda de informações importantes.  
- Adotei uma **abordagem robusta de normalização**, reduzindo o impacto dos outliers sem eliminar dados válidos.  
- **Dividi o dataset em treino e teste.**  
- Gereii **versões normalizadas** para treino e teste usando o **RobustScaler**, que utiliza a **mediana** e o **intervalo interquartil (IQR)** para escalar as variáveis contínuas.  

---

📌 **Resumo:**  
Este trabalho teve como objetivo preparar o *dataset Heart Disease (Cleveland)* para análises e modelagem preditiva, aplicando técnicas de exploração, detecção de outliers e normalização robusta, de forma a garantir a integridade e qualidade dos dados.
