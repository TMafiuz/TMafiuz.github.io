# Modelo KNN para Dataset Titanic
# Clasificación binaria de supervivencia usando KNeighborsClassifier

# Importaciones
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

print("=" * 50)
print("MODELO KNN PARA SUPERVIVENCIA DEL TITANIC")
print("=" * 50)

# 1. Carga y separación de datos
print("\n1. Cargando datos...")
train_df = pd.read_csv('Taller 2/titanic/train.csv')
test_df = pd.read_csv('Taller 2/titanic/test.csv')

print(f"Datos de entrenamiento: {train_df.shape}")
print(f"Datos de prueba: {test_df.shape}")

# Separar características y objetivo del conjunto de entrenamiento
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_df['Survived']

# Características de prueba
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

print(f"Características (X): {X.shape}")
print(f"Objetivo (y): {y.shape}")
print(f"Características de prueba: {X_test.shape}")

# 2. Definición de características numéricas y categóricas
num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Sex', 'Embarked']

print(f"\nCaracterísticas numéricas: {num_features}")
print(f"Características categóricas: {cat_features}")

# 3. Construcción del preprocesador
print("\n2. Configurando preprocesamiento...")

# Transformador para características numéricas: imputación (mediana) + StandardScaler
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformador para características categóricas: imputación (moda) + OneHotEncoder
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer que combina ambos transformadores
preprocess = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# 4. Modelo KNN
print("3. Configurando modelo KNN...")
model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',
    p=2  # euclidiana
)

# 5. Pipeline completo
pipe = Pipeline([
    ('prep', preprocess),
    ('clf', model)
])

print("Pipeline configurado correctamente.")

# 6. Partición estratificada
print("\n4. Dividiendo datos en entrenamiento y validación...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y, 
    random_state=42
)

print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de validación: {X_valid.shape}")
print(f"Distribución en entrenamiento - Sobrevivieron: {y_train.sum()}/{len(y_train)} ({y_train.mean():.3f})")
print(f"Distribución en validación - Sobrevivieron: {y_valid.sum()}/{len(y_valid)} ({y_valid.mean():.3f})")

# 7. Entrenamiento
print("\n5. Entrenando modelo...")
pipe.fit(X_train, y_train)

# 8. Predicción en validación
print("6. Realizando predicciones en validación...")
y_pred = pipe.predict(X_valid)

# 9. Métricas de evaluación
print("\n" + "=" * 50)
print("RESULTADOS DE VALIDACIÓN")
print("=" * 50)

# Matriz de confusión
cm = confusion_matrix(y_valid, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print(f"\nMatriz de Confusión:")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")

# Métricas principales
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_valid, y_pred)

print(f"\nMÉTRICAS DE RENDIMIENTO:")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Sensibilidad (Recall): {recall:.4f}")
print(f"Especificidad: {specificity:.4f}")
print(f"F1-Score: {f1:.4f}")

# Reporte de clasificación completo
print(f"\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_valid, y_pred))

# Mostrar matriz de confusión gráficamente
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Sobrevivió', 'Sobrevivió'])
disp.plot()
plt.title('Matriz de Confusión - Modelo KNN Titanic')
plt.tight_layout()
plt.savefig('confusion_matrix_knn_titanic.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Re-entrenar en todo el conjunto de entrenamiento y predecir en test
print("\n7. Re-entrenando en todo el conjunto de entrenamiento...")
pipe.fit(X, y)

print("8. Prediciendo en conjunto de prueba...")
test_predictions = pipe.predict(X_test)

# 11. Guardar resultados
print("\n9. Guardando resultados...")

# Crear dataframe con las predicciones
results_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# Guardar predicciones en formato CSV
results_df.to_csv('titanic_knn_predictions.csv', index=False)

# Guardar resumen completo en archivo de texto
with open('resultados_knn_titanic.txt', 'w', encoding='utf-8') as f:
    f.write("RESULTADOS MODELO KNN - DATASET TITANIC\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("CONFIGURACIÓN DEL MODELO:\n")
    f.write(f"- Algoritmo: K-Nearest Neighbors (KNN)\n")
    f.write(f"- n_neighbors: 5\n")
    f.write(f"- weights: uniform\n")
    f.write(f"- metric: minkowski (p=2, euclidiana)\n")
    f.write(f"- Características utilizadas: {num_features + cat_features}\n")
    f.write(f"- Preprocesamiento: Imputación + Estandarización + One-Hot Encoding\n\n")
    
    f.write("PARTICIÓN DE DATOS:\n")
    f.write(f"- Conjunto de entrenamiento: {X_train.shape[0]} muestras\n")
    f.write(f"- Conjunto de validación: {X_valid.shape[0]} muestras\n")
    f.write(f"- Test size: 0.3, stratified, random_state=42\n\n")
    
    f.write("MATRIZ DE CONFUSIÓN (Validación):\n")
    f.write(f"TN: {tn}, FP: {fp}\n")
    f.write(f"FN: {fn}, TP: {tp}\n\n")
    
    f.write("MÉTRICAS DE RENDIMIENTO (Validación):\n")
    f.write(f"Exactitud (Accuracy): {accuracy:.4f}\n")
    f.write(f"Precisión (Precision): {precision:.4f}\n")
    f.write(f"Sensibilidad (Recall): {recall:.4f}\n")
    f.write(f"Especificidad: {specificity:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")
    
    f.write("REPORTE DE CLASIFICACIÓN:\n")
    f.write(classification_report(y_valid, y_pred))
    f.write(f"\n\nPREDICCIONES EN CONJUNTO DE PRUEBA:\n")
    f.write(f"Total de predicciones: {len(test_predictions)}\n")
    f.write(f"Sobrevivientes predichos: {test_predictions.sum()}\n")
    f.write(f"No sobrevivientes predichos: {len(test_predictions) - test_predictions.sum()}\n")
    f.write(f"Tasa de supervivencia predicha: {test_predictions.mean():.3f}\n")

print(f"\nArchivos generados:")
print(f"- titanic_knn_predictions.csv: Predicciones para el conjunto de prueba")
print(f"- resultados_knn_titanic.txt: Resumen completo de resultados")
print(f"- confusion_matrix_knn_titanic.png: Gráfico de matriz de confusión")

print("\n" + "=" * 50)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 50)