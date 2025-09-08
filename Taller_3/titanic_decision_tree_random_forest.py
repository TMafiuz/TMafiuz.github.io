"""
ANÁLISIS DEL DATASET TITANIC CON ÁRBOLES DE DECISIÓN Y RANDOM FOREST

Objetivo: Predecir la supervivencia de pasajeros del Titanic utilizando
modelos de Árbol de Decisión y Random Forest.

Autor: Análisis de Machine Learning
Dataset: Titanic - Clasificación Binaria (Supervivencia: 0 o 1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS DEL DATASET TITANIC")
print("Árboles de Decisión vs Random Forest")
print("=" * 80)


# Cargar el dataset
df = pd.read_csv('Taller_3\Titanic\Titanic-Dataset.csv')

print(f"Dimensiones del dataset: {df.shape}")
print(f"Variables disponibles: {list(df.columns)}")
print(f"\nPrimeras 5 filas:")
print(df.head())

print(f"\nInformación sobre valores faltantes:")
print(df.isnull().sum())

print(f"\nDistribución de la variable objetivo (Survived):")
print(df['Survived'].value_counts())
print(f"Tasa de supervivencia: {df['Survived'].mean():.2%}")

# 3. PREPROCESAMIENTO DE DATOS
print("\n3. PREPROCESAMIENTO DE DATOS")
print("-" * 40)

# Crear una copia para el preprocesamiento
data = df.copy()

# Seleccionar características relevantes
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Manejo de valores faltantes
print("Manejo de valores faltantes:")
print("- Age: Imputar con la mediana")
print("- Embarked: Imputar con la moda")
print("- Fare: Imputar con la mediana")

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Codificación de variables categóricas
print("\nCodificación de variables categóricas:")
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

data['Sex_encoded'] = le_sex.fit_transform(data['Sex'])
data['Embarked_encoded'] = le_embarked.fit_transform(data['Embarked'])

# Seleccionar características finales
features_final = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = data[features_final]
y = data[target]

print(f"Características finales: {features_final}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisión de datos:")
print(f"- Entrenamiento: {X_train.shape[0]} muestras")
print(f"- Prueba: {X_test.shape[0]} muestras")

# 4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS
print("\n4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS")
print("-" * 40)

# 4.1 Árbol de Decisión
print("\n4.1 ÁRBOL DE DECISIÓN - CONFIGURACIÓN DETALLADA")
print("=" * 50)
print("CONCEPTO: Un árbol de decisión es un modelo que toma decisiones siguiendo")
print("una serie de preguntas binarias organizadas en forma de árbol.")
print()
print("CONFIGURACIÓN DE HIPERPARÁMETROS:")
print("┌─ max_depth=5:")
print("│  • Limita la profundidad máxima del árbol a 5 niveles")
print("│  • Previene overfitting al evitar árboles muy complejos")
print("│  • Valor elegido tras experimentar con 3, 5, 7, 10")
print("│")
print("┌─ min_samples_split=20:")
print("│  • Un nodo debe tener al menos 20 muestras para dividirse")
print("│  • Evita divisiones con muy pocos datos")
print("│  • Mejora la generalización del modelo")
print("│")
print("┌─ min_samples_leaf=10:")
print("│  • Cada hoja debe contener al menos 10 muestras")
print("│  • Previene hojas con decisiones basadas en muy pocos casos")
print("│  • Reduce la varianza del modelo")
print("│")
print("└─ random_state=42: Garantiza reproducibilidad de resultados")
print()

dt_classifier = DecisionTreeClassifier(
    max_depth=5,           
    min_samples_split=20,  
    min_samples_leaf=10,   
    random_state=42
)

print("Entrenando Árbol de Decisión...")
dt_classifier.fit(X_train, y_train)
print("✓ Árbol de Decisión entrenado exitosamente")
print(f"  • Profundidad máxima configurada: {dt_classifier.max_depth}")
print(f"  • Profundidad real alcanzada: {dt_classifier.get_depth()}")
print(f"  • Número de hojas: {dt_classifier.get_n_leaves()}")
print(f"  • Muestras mínimas para dividir: {dt_classifier.min_samples_split}")

# 4.2 Random Forest
print("\n4.2 RANDOM FOREST - CONFIGURACIÓN DETALLADA")
print("=" * 50)
print("CONCEPTO: Random Forest es un ENSEMBLE (conjunto) de múltiples árboles")
print("de decisión que trabajan juntos para hacer predicciones más robustas.")
print()
print("¿CÓMO FUNCIONA EL ENSEMBLE?")
print("┌─ Bootstrapping:")
print("│  • Cada árbol se entrena con una muestra aleatoria de los datos")
print("│  • Aproximadamente 63% de los datos originales por árbol")
print("│  • Permite que cada árbol 'vea' datos ligeramente diferentes")
print("│")
print("┌─ Random Feature Selection:")
print("│  • En cada división, solo considera un subconjunto aleatorio de características")
print("│  • Reduce la correlación entre árboles individuales")
print("│  • Mejora la diversidad del ensemble")
print("│")
print("└─ Voting/Promedio:")
print("   • Para clasificación: voto mayoritario de todos los árboles")
print("   • Suaviza las predicciones y reduce overfitting")
print()
print("CONFIGURACIÓN DE HIPERPARÁMETROS:")
print("┌─ n_estimators=100:")
print("│  • Número de árboles en el bosque")
print("│  • Más árboles = mayor estabilidad, pero más tiempo de cómputo")
print("│  • 100 es un balance entre rendimiento y eficiencia")
print("│")
print("┌─ max_depth=5:")
print("│  • Profundidad máxima de cada árbol individual")
print("│  • Mismo valor que el árbol simple para comparación justa")
print("│  • Cada árbol es relativamente 'débil' individualmente")
print("│")
print("┌─ min_samples_split=20 y min_samples_leaf=10:")
print("│  • Mismos valores que el árbol individual")
print("│  • Controlan la complejidad de cada árbol del ensemble")
print("│")
print("└─ random_state=42: Reproducibilidad del proceso aleatorio")
print()

rf_classifier = RandomForestClassifier(
    n_estimators=100,      
    max_depth=5,           
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1              # Usar todos los cores disponibles para acelerar
)

print("Entrenando Random Forest (Ensemble de 100 árboles)...")
rf_classifier.fit(X_train, y_train)
print("✓ Random Forest entrenado exitosamente")
print(f"  • Número de árboles (n_estimators): {rf_classifier.n_estimators}")
print(f"  • Profundidad máxima por árbol: {rf_classifier.max_depth}")
print(f"  • Características consideradas por división: √{len(features_final)} ≈ {int(np.sqrt(len(features_final)))}")
print(f"  • Muestras de bootstrap por árbol: ~{int(0.632 * len(X_train))}")
print(f"  • Procesamiento paralelo: {rf_classifier.n_jobs} cores")

print("\n🔍 VENTAJAS DEL ENSEMBLE:")
print("• Reduce overfitting comparado con un árbol individual")
print("• Mayor robustez ante datos ruidosos o outliers") 
print("• Mejor generalización en datos no vistos")
print("• Proporciona medidas de importancia de características más estables")
print("• Maneja bien la no linealidad en los datos")

# 5. EVALUACIÓN Y COMPARACIÓN DE MODELOS
print("\n5. EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("-" * 40)

# Predicciones
dt_pred = dt_classifier.predict(X_test)
dt_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

rf_pred = rf_classifier.predict(X_test)
rf_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Función para calcular métricas
def calcular_metricas(y_true, y_pred, y_pred_proba, modelo_nombre):
    print(f"\nMÉTRICAS PARA {modelo_nombre.upper()}")
    print("-" * 30)
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Precision): {precision:.4f}")
    print(f"Exhaustividad (Recall): {recall:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de Confusión:")
    print(f"   Predicho:  No  Sí")
    print(f"Real No:    {cm[0,0]:3d} {cm[0,1]:3d}")
    print(f"     Sí:    {cm[1,0]:3d} {cm[1,1]:3d}")
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 
        'f1': f1, 'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'cm': cm
    }

# Calcular métricas para ambos modelos
dt_metrics = calcular_metricas(y_test, dt_pred, dt_pred_proba, "Árbol de Decisión")
rf_metrics = calcular_metricas(y_test, rf_pred, rf_pred_proba, "Random Forest")

# 6. ANÁLISIS COMPARATIVO
print("\n6. ANÁLISIS COMPARATIVO")
print("-" * 40)

# Comparación de métricas
metricas_comparacion = pd.DataFrame({
    'Árbol de Decisión': [dt_metrics['accuracy'], dt_metrics['precision'], 
                         dt_metrics['recall'], dt_metrics['f1'], dt_metrics['auc']],
    'Random Forest': [rf_metrics['accuracy'], rf_metrics['precision'], 
                     rf_metrics['recall'], rf_metrics['f1'], rf_metrics['auc']]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])

print("Comparación de Métricas:")
print(metricas_comparacion.round(4))

# Determinar el mejor modelo
mejor_modelo = 'Random Forest' if rf_metrics['f1'] > dt_metrics['f1'] else 'Árbol de Decisión'
print(f"\nMejor modelo basado en F1-Score: {mejor_modelo}")

# 7. VISUALIZACIONES
print("\n7. GENERANDO VISUALIZACIONES")
print("-" * 40)

# Configurar subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis Titanic: Árboles de Decisión vs Random Forest', fontsize=16)

# 7.1 Matrices de Confusión
# Árbol de Decisión
sns.heatmap(dt_metrics['cm'], annot=True, fmt='d', cmap='Blues', 
           xticklabels=['No Sobrevivió', 'Sobrevivió'],
           yticklabels=['No Sobrevivió', 'Sobrevivió'], ax=axes[0,0])
axes[0,0].set_title('Matriz de Confusión - Árbol de Decisión')
axes[0,0].set_ylabel('Valores Reales')
axes[0,0].set_xlabel('Valores Predichos')

# Random Forest
sns.heatmap(rf_metrics['cm'], annot=True, fmt='d', cmap='Greens',
           xticklabels=['No Sobrevivió', 'Sobrevivió'],
           yticklabels=['No Sobrevivió', 'Sobrevivió'], ax=axes[0,1])
axes[0,1].set_title('Matriz de Confusión - Random Forest')
axes[0,1].set_ylabel('Valores Reales')
axes[0,1].set_xlabel('Valores Predichos')

# 7.2 Comparación de Métricas
metricas_comparacion.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Comparación de Métricas')
axes[1,0].set_ylabel('Puntuación')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].legend()

# 7.3 Curvas ROC
axes[1,1].plot(dt_metrics['fpr'], dt_metrics['tpr'], 
              label=f'Árbol de Decisión (AUC = {dt_metrics["auc"]:.3f})', linewidth=2)
axes[1,1].plot(rf_metrics['fpr'], rf_metrics['tpr'], 
              label=f'Random Forest (AUC = {rf_metrics["auc"]:.3f})', linewidth=2)
axes[1,1].plot([0, 1], [0, 1], 'k--', label='Línea Base')
axes[1,1].set_xlabel('Tasa de Falsos Positivos')
axes[1,1].set_ylabel('Tasa de Verdaderos Positivos')
axes[1,1].set_title('Curvas ROC')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('titanic_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. IMPORTANCIA DE CARACTERÍSTICAS (SOLO RANDOM FOREST)
print("\n8. IMPORTANCIA DE CARACTERÍSTICAS")
print("-" * 40)

importancias = rf_classifier.feature_importances_
indices_importancia = np.argsort(importancias)[::-1]

print("Ranking de importancia de características (Random Forest):")
for i in range(len(features_final)):
    print(f"{i+1}. {features_final[indices_importancia[i]]}: "
          f"{importancias[indices_importancia[i]]:.4f}")

# Visualización de importancia
plt.figure(figsize=(10, 6))
plt.bar(range(len(importancias)), importancias[indices_importancia])
plt.xticks(range(len(importancias)), 
          [features_final[i] for i in indices_importancia], rotation=45)
plt.title('Importancia de Características - Random Forest')
plt.ylabel('Importancia')
plt.tight_layout()
plt.savefig('titanic_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


print(f"\n✓ Análisis completado. Archivos guardados:")
print("  - titanic_analysis_results.png")
print("  - titanic_feature_importance.png")
