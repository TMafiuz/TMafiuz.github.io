

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS DEL DATASET IRIS")
print("Árboles de Decisión vs Random Forest")
print("=" * 80)


# Cargar el dataset
df = pd.read_csv('Taller_3\Iris\Iris.csv')

print(f"Dimensiones del dataset: {df.shape}")
print(f"Variables disponibles: {list(df.columns)}")
print(f"\nPrimeras 5 filas:")
print(df.head())

print(f"\nInformación sobre valores faltantes:")
print(df.isnull().sum())

print(f"\nDistribución de las especies (variable objetivo):")
species_counts = df['Species'].value_counts()
print(species_counts)

# Estadísticas descriptivas
print(f"\nEstadísticas descriptivas de las características:")
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
print(df[numeric_cols].describe())

print("\n3. PREPROCESAMIENTO DE DATOS")
print("-" * 40)

# Crear una copia para el preprocesamiento
data = df.copy()

# Seleccionar características (excluir Id)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = 'Species'

X = data[features]
y = data[target]

print(f"Características utilizadas: {features}")
print(f"Variable objetivo: {target}")
print(f"Forma de X: {X.shape}")
print(f"Clases únicas: {y.unique()}")

# Codificar las etiquetas de especies
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nCodificación de especies:")
for i, species in enumerate(le.classes_):
    print(f"  {species}: {i}")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nDivisión de datos:")
print(f"- Entrenamiento: {X_train.shape[0]} muestras")
print(f"- Prueba: {X_test.shape[0]} muestras")

# Distribución en conjuntos de entrenamiento y prueba
print(f"\nDistribución en entrenamiento: {np.bincount(y_train)}")
print(f"Distribución en prueba: {np.bincount(y_test)}")

# 4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS
print("\n4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS")
print("-" * 40)

# 4.1 Árbol de Decisión
print("\n4.1 ÁRBOL DE DECISIÓN - CONFIGURACIÓN PARA MULTICLASE")
print("=" * 55)
print("CONCEPTO: Árbol de decisión para clasificación multiclase (3 especies de Iris)")
print("El árbol hace preguntas sobre las características de las flores para")
print("clasificarlas en una de las tres especies.")
print()
print("CONFIGURACIÓN DE HIPERPARÁMETROS PARA IRIS:")
print("┌─ max_depth=4:")
print("│  • Profundidad máxima reducida (dataset pequeño y simple)")
print("│  • Iris es un problema relativamente fácil de clasificar")
print("│  • Previene overfitting en un dataset de solo 150 muestras")
print("│")
print("┌─ min_samples_split=5:")
print("│  • Reducido vs Titanic debido al menor tamaño del dataset")
print("│  • Un nodo necesita al menos 5 muestras para dividirse")
print("│  • Apropiado para dataset con 105 muestras de entrenamiento")
print("│")
print("┌─ min_samples_leaf=2:")
print("│  • Cada hoja debe tener al menos 2 muestras")
print("│  • Valor bajo apropiado para dataset pequeño pero balanceado")
print("│  • Permite capturar patrones específicos de cada especie")
print("│")
print("└─ random_state=42: Reproducibilidad de resultados")
print()
print("ESTRATEGIA MULTICLASE:")
print("• El árbol maneja naturalmente clasificación multiclase")
print("• Cada división busca la mejor separación entre las 3 clases")
print("• Las hojas contendrán muestras mayoritariamente de una especie")
print()

dt_classifier = DecisionTreeClassifier(
    max_depth=4,           
    min_samples_split=5,   
    min_samples_leaf=2,    
    random_state=42
)

print("Entrenando Árbol de Decisión para clasificación multiclase...")
dt_classifier.fit(X_train, y_train)
print("✓ Árbol de Decisión entrenado exitosamente")
print(f"  • Profundidad máxima configurada: {dt_classifier.max_depth}")
print(f"  • Profundidad real alcanzada: {dt_classifier.get_depth()}")
print(f"  • Número de hojas: {dt_classifier.get_n_leaves()}")
print(f"  • Clases manejadas: {len(dt_classifier.classes_)} especies")

# 4.2 Random Forest
print("\n4.2 RANDOM FOREST - ENSEMBLE PARA MULTICLASE")
print("=" * 55)
print("CONCEPTO: Ensemble de 100 árboles trabajando juntos para clasificar")
print("las especies de Iris. Cada árbol 'vota' por una especie y gana la mayoría.")
print()
print("VENTAJAS DEL ENSEMBLE EN MULTICLASE:")
print("┌─ Robustez:")
print("│  • Si un árbol se confunde entre dos especies similares,")
print("│  • otros árboles pueden corregir el error")
print("│  • Reduce la variabilidad en las predicciones")
print("│")
print("┌─ Mejor separación de clases:")
print("│  • Diferentes árboles pueden especializarse en separar")
print("│  • pares específicos de especies (setosa vs resto, etc.)")
print("│  • Captura mejor la complejidad de los límites de decisión")
print("│")
print("└─ Estimación de confianza:")
print("   • El porcentaje de votos indica qué tan 'seguro' está el modelo")
print("   • Útil para identificar casos ambiguos entre especies")
print()
print("CONFIGURACIÓN DE HIPERPARÁMETROS:")
print("┌─ n_estimators=100:")
print("│  • 100 árboles en el bosque")
print("│  • Número estándar que balancea rendimiento y eficiencia")
print("│  • Cada árbol entrena con ~67 muestras (bootstrap)")
print("│")
print("┌─ max_depth=4:")
print("│  • Cada árbol individual es relativamente simple")
print("│  • La complejidad surge del ensemble, no de árboles profundos")
print("│  • Apropiado para el dataset Iris")
print("│")
print("┌─ min_samples_split=5 y min_samples_leaf=2:")
print("│  • Parámetros ajustados al tamaño del dataset")
print("│  • Permiten que cada árbol capture patrones específicos")
print("│  • Evitan overfitting manteniendo la simplicidad")
print("│")
print("└─ random_state=42: Control de la aleatoriedad del ensemble")
print()

rf_classifier = RandomForestClassifier(
    n_estimators=100,      
    max_depth=4,           
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Entrenando Random Forest (100 árboles para 3 especies)...")
rf_classifier.fit(X_train, y_train)
print("✓ Random Forest entrenado exitosamente")
print(f"  • Número de árboles: {rf_classifier.n_estimators}")
print(f"  • Profundidad máxima por árbol: {rf_classifier.max_depth}")
print(f"  • Características consideradas por división: √{len(X_train.columns)} ≈ {int(np.sqrt(len(X_train.columns)))}")
print(f"  • Muestras de entrenamiento por árbol: ~{int(0.632 * len(X_train))}")
print(f"  • Especies clasificadas: {len(rf_classifier.classes_)}")

print("\n🌸 ESPECIALIZACIÓN PARA CLASIFICACIÓN DE IRIS:")
print("• Setosa: Generalmente fácil de separar (características distintivas)")
print("• Versicolor vs Virginica: Más desafiante, aquí brilla el ensemble")
print("• El Random Forest puede crear múltiples estrategias de separación")
print("• Cada árbol puede usar diferentes combinaciones de características")

# 5. EVALUACIÓN Y COMPARACIÓN DE MODELOS
print("\n5. EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("-" * 40)

# Predicciones
dt_pred = dt_classifier.predict(X_test)
dt_pred_proba = dt_classifier.predict_proba(X_test)

rf_pred = rf_classifier.predict(X_test)
rf_pred_proba = rf_classifier.predict_proba(X_test)

# Función para calcular métricas multiclase
def calcular_metricas_multiclase(y_true, y_pred, y_pred_proba, modelo_nombre, class_names):
    print(f"\nMÉTRICAS PARA {modelo_nombre.upper()}")
    print("-" * 30)
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"\nMétricas Macro (promedio no ponderado):")
    print(f"  Precisión (Precision): {precision_macro:.4f}")
    print(f"  Exhaustividad (Recall): {recall_macro:.4f}")
    print(f"  Puntuación F1: {f1_macro:.4f}")
    
    print(f"\nMétricas Micro (promedio ponderado):")
    print(f"  Precisión (Precision): {precision_micro:.4f}")
    print(f"  Exhaustividad (Recall): {recall_micro:.4f}")
    print(f"  Puntuación F1: {f1_micro:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de Confusión:")
    print("           Predicho")
    print(f"           {class_names[0][:4]:>6} {class_names[1][:4]:>6} {class_names[2][:4]:>6}")
    for i, real_class in enumerate(class_names):
        print(f"Real {real_class[:4]:>4}: {cm[i,0]:6d} {cm[i,1]:6d} {cm[i,2]:6d}")
    
    # Reporte de clasificación por clase
    print(f"\nReporte detallado por clase:")
    class_report = classification_report(y_true, y_pred, target_names=class_names, 
                                       output_dict=True)
    for i, class_name in enumerate(class_names):
        metrics = class_report[class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")
    
    return {
        'accuracy': accuracy, 'precision_macro': precision_macro, 
        'recall_macro': recall_macro, 'f1_macro': f1_macro,
        'precision_micro': precision_micro, 'recall_micro': recall_micro, 
        'f1_micro': f1_micro, 'cm': cm, 'class_report': class_report,
        'pred_proba': y_pred_proba
    }

# Nombres de las clases
class_names = le.classes_

# Calcular métricas para ambos modelos
dt_metrics = calcular_metricas_multiclase(y_test, dt_pred, dt_pred_proba, 
                                        "Árbol de Decisión", class_names)
rf_metrics = calcular_metricas_multiclase(y_test, rf_pred, rf_pred_proba, 
                                        "Random Forest", class_names)

# 6. ANÁLISIS COMPARATIVO
print("\n6. ANÁLISIS COMPARATIVO")
print("-" * 40)

# Comparación de métricas
metricas_comparacion = pd.DataFrame({
    'Árbol de Decisión': [
        dt_metrics['accuracy'], dt_metrics['precision_macro'], 
        dt_metrics['recall_macro'], dt_metrics['f1_macro']
    ],
    'Random Forest': [
        rf_metrics['accuracy'], rf_metrics['precision_macro'], 
        rf_metrics['recall_macro'], rf_metrics['f1_macro']
    ]
}, index=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'])

print("Comparación de Métricas:")
print(metricas_comparacion.round(4))

# Determinar el mejor modelo
mejor_modelo = 'Random Forest' if rf_metrics['f1_macro'] > dt_metrics['f1_macro'] else 'Árbol de Decisión'
print(f"\nMejor modelo basado en F1-Score Macro: {mejor_modelo}")

# 7. VISUALIZACIONES
print("\n7. GENERANDO VISUALIZACIONES")
print("-" * 40)

# Configurar subplots
fig = plt.figure(figsize=(20, 15))

# 7.1 Distribución de datos por especies (subplot 1)
plt.subplot(3, 3, 1)
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.title('Distribución: Sépalo Longitud vs Ancho')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(3, 3, 2)
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')
plt.title('Distribución: Pétalo Longitud vs Ancho')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 7.2 Matrices de Confusión
plt.subplot(3, 3, 4)
sns.heatmap(dt_metrics['cm'], annot=True, fmt='d', cmap='Blues',
           xticklabels=[name[:4] for name in class_names],
           yticklabels=[name[:4] for name in class_names])
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.ylabel('Valores Reales')
plt.xlabel('Valores Predichos')

plt.subplot(3, 3, 5)
sns.heatmap(rf_metrics['cm'], annot=True, fmt='d', cmap='Greens',
           xticklabels=[name[:4] for name in class_names],
           yticklabels=[name[:4] for name in class_names])
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Valores Reales')
plt.xlabel('Valores Predichos')

# 7.3 Comparación de Métricas
plt.subplot(3, 3, 6)
metricas_comparacion.plot(kind='bar', ax=plt.gca())
plt.title('Comparación de Métricas')
plt.ylabel('Puntuación')
plt.xticks(rotation=45)
plt.legend()

# 7.4 ROC Curves para clasificación multiclase (One-vs-Rest)
# Binarizar las etiquetas para ROC multiclase
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

plt.subplot(3, 3, 7)
colors = ['blue', 'red', 'green']
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], dt_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('ROC - Árbol de Decisión')
plt.legend(loc="lower right")

plt.subplot(3, 3, 8)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('ROC - Random Forest')
plt.legend(loc="lower right")

# 7.5 Importancia de características (Random Forest)
plt.subplot(3, 3, 9)
importancias = rf_classifier.feature_importances_
indices = np.argsort(importancias)[::-1]
plt.bar(range(len(importancias)), importancias[indices])
plt.xticks(range(len(importancias)), [features[i] for i in indices], rotation=45)
plt.title('Importancia de Características - RF')
plt.ylabel('Importancia')

plt.suptitle('Análisis Iris: Árboles de Decisión vs Random Forest', fontsize=16)
plt.tight_layout()
plt.savefig('iris_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
print("\n8. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN")
print("-" * 40)

plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=features,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Árbol de Decisión - Dataset Iris')
plt.savefig('iris_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. IMPORTANCIA DE CARACTERÍSTICAS DETALLADA
print("\n9. IMPORTANCIA DE CARACTERÍSTICAS")
print("-" * 40)

importancias = rf_classifier.feature_importances_
indices_importancia = np.argsort(importancias)[::-1]

print("Ranking de importancia de características (Random Forest):")
for i in range(len(features)):
    print(f"{i+1}. {features[indices_importancia[i]]}: "
          f"{importancias[indices_importancia[i]]:.4f}")

# Comparar importancias entre modelos
print("\nComparación de importancias:")
dt_importancias = dt_classifier.feature_importances_
rf_importancias = rf_classifier.feature_importances_

importancia_df = pd.DataFrame({
    'Característica': features,
    'Árbol de Decisión': dt_importancias,
    'Random Forest': rf_importancias
})
print(importancia_df.round(4))


# Análisis específico de rendimiento
if dt_metrics['accuracy'] > 0.95 and rf_metrics['accuracy'] > 0.95:
    print("• Ambos modelos muestran excelente rendimiento en este dataset")
elif rf_metrics['accuracy'] > dt_metrics['accuracy']:
    print("• Random Forest supera al Árbol de Decisión en este caso")
else:
    print("• El Árbol de Decisión muestra mejor rendimiento")

print(f"\n✓ Análisis completado. Archivos guardados:")
print("  - iris_analysis_results.png")
print("  - iris_decision_tree.png")
