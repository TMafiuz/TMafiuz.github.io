# Análisis de Machine Learning: Árboles de Decisión vs Random Forest

Este proyecto implementa tres análisis completos de Machine Learning comparando el rendimiento de Árboles de Decisión y Random Forest en diferentes tipos de problemas de clasificación.

## 📋 Datasets Analizados

1. **Titanic** - Supervivencia de pasajeros (Clasificación Binaria)
2. **Iris** - Clasificación de especies de flores (Clasificación Multiclase)
3. **Detección de Fraude** - Transacciones fraudulentas (Clasificación Binaria Desbalanceada)

## 🚀 Cómo Ejecutar

### Prerrequisitos

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecución de los Scripts

1. **Análisis de Titanic:**
```bash
cd "Taller 3"
python titanic_decision_tree_random_forest.py
```

2. **Análisis de Iris:**
```bash
cd "Taller 3"
python iris_decision_tree_random_forest.py
```

3. **Análisis de Detección de Fraude:**
```bash
cd "Taller 3"
python fraud_detection_decision_tree_random_forest.py
```

## 📊 Estructura del Análisis

Cada script sigue la misma estructura metodológica:

### 1. Introducción (10%)
- Presentación del dataset
- Definición del problema de clasificación
- Justificación y desafíos esperados

### 2. Preprocesamiento de Datos (20%)
- **Manejo de valores faltantes:** Imputación con mediana/moda
- **Codificación de variables categóricas:** LabelEncoder para variables binarias
- **División de datos:** 80% entrenamiento, 20% prueba con estratificación
- **Características derivadas:** Creación de features adicionales cuando es relevante

### 3. Construcción y Entrenamiento de Modelos (20%)

#### Árbol de Decisión:
- Configuración de hiperparámetros (max_depth, min_samples_split, min_samples_leaf)
- Balance de clases cuando es necesario
- Explicación de la configuración elegida

#### Random Forest:
- Ensamble de 100 árboles
- Mismos hiperparámetros base que el árbol individual
- Uso de todos los cores disponibles (n_jobs=-1)

### 4. Evaluación y Comparación de Modelos (30%)

#### Métricas de Clasificación Calculadas:

**Para todos los datasets:**
- **Accuracy:** Precisión general del modelo
- **Precision:** Proporción de predicciones positivas correctas
- **Recall (Sensibilidad):** Proporción de casos positivos detectados
- **F1-Score:** Media armónica entre Precision y Recall

**Visualizaciones incluidas:**
- **Matriz de Confusión:** Para ambos modelos con interpretación detallada
- **Curvas ROC:** Con valores de AUC para comparación
- **Gráficos de barras:** Comparación directa de métricas
- **Importancia de características:** Ranking de features más relevantes

**Análisis específicos por dataset:**

- **Titanic:** Enfoque en supervivencia con análisis demográfico
- **Iris:** Clasificación multiclase con ROC One-vs-Rest
- **Fraude:** Énfasis especial en Recall (crítico para detectar fraudes)

### 5. Análisis Comparativo Detallado

- Comparación tabular de todas las métricas
- Identificación del mejor modelo por F1-Score
- Análisis de fortalezas y debilidades de cada enfoque
- Recomendaciones específicas para cada problema

## 📈 Outputs Generados

Cada script genera:

### Archivos de Imagen:
- `{dataset}_analysis_results.png` - Análisis completo con múltiples gráficos
- `{dataset}_feature_importance.png` - Importancia de características
- `iris_decision_tree.png` - Visualización del árbol (solo Iris)

### Reportes en Consola:
- Análisis paso a paso con métricas detalladas
- Interpretación de matrices de confusión
- Ranking de importancia de características
- Conclusiones y recomendaciones

## 🎯 Características Especiales por Dataset

### Titanic:
- Manejo de datos faltantes en Age, Embarked, Fare
- Codificación de variables categóricas (Sex, Embarked)
- Análisis de supervivencia por género y clase social

### Iris:
- Problema de clasificación "perfecto" para demostrar capacidades
- Visualización completa del árbol de decisión
- ROC multiclase con One-vs-Rest
- Análisis de separabilidad de especies

### Detección de Fraude:
- **Énfasis en Recall:** Crítico para no perder fraudes
- **Balance de clases:** Manejo de dataset desbalanceado
- **Análisis de costo-beneficio:** Evaluación del costo de errores
- **Análisis de umbral óptimo:** Para optimizar detección
- **Características derivadas:** AmountZScore, RiskScore, etc.

## 📝 Métricas de Evaluación Detalladas

### Matriz de Confusión (Para clasificación binaria):
```
                Predicho
              No    Sí
Real    No   TN    FP
        Sí   FN    TP
```

- **TN (True Negatives):** Casos negativos correctamente clasificados
- **TP (True Positives):** Casos positivos correctamente clasificados
- **FP (False Positives):** Falsos positivos (Tipo I)
- **FN (False Negatives):** Falsos negativos (Tipo II)

### Fórmulas de Métricas:
- **Accuracy = (TP + TN) / (TP + TN + FP + FN)**
- **Precision = TP / (TP + FP)**
- **Recall = TP / (TP + FN)**
- **F1-Score = 2 × (Precision × Recall) / (Precision + Recall)**
- **Specificity = TN / (TN + FP)**

## 🏆 Resultados Esperados

### Rendimiento General:
- **Random Forest** típicamente supera a **Árbol de Decisión** individual
- **RF** es más robusto al overfitting
- **Árbol individual** es más interpretable

### Por Dataset:
- **Iris:** Ambos modelos logran >95% accuracy (dataset fácil)
- **Titanic:** RF ~82-85% accuracy vs DT ~80-83%
- **Fraude:** Énfasis en maximizar Recall (>90% deseable)

## 🛠️ Configuración de Hiperparámetros

### Árbol de Decisión:
```python
DecisionTreeClassifier(
    max_depth=4-8,         # Según complejidad del dataset
    min_samples_split=5-20, # Prevenir overfitting
    min_samples_leaf=2-10,  # Asegurar hojas significativas
    class_weight='balanced' # Para datasets desbalanceados
)
```

### Random Forest:
```python
RandomForestClassifier(
    n_estimators=100,       # Número de árboles
    max_depth=4-8,          # Igual que árbol individual
    min_samples_split=5-20, # Prevenir overfitting
    min_samples_leaf=2-10,  # Asegurar hojas significativas
    class_weight='balanced' # Para datasets desbalanceados
)
```

## 📚 Conceptos Clave Implementados

### Árboles de Decisión:
- Algoritmo de partición recursiva
- Criterios de división (Gini, Entropy)
- Podado para prevenir overfitting
- Interpretabilidad alta

### Random Forest:
- Ensamble de árboles (Bootstrap Aggregating)
- Selección aleatoria de características
- Votación por mayoría
- Reducción de varianza

### Técnicas de Evaluación:
- Validación cruzada estratificada
- Curvas ROC y AUC
- Análisis de importancia de características
- Métricas específicas por dominio

## 🚨 Consideraciones Especiales

### Para Detección de Fraude:
- **Recall es más importante que Precision**
- Costo asimétrico de errores (FN >> FP)
- Necesidad de ajuste de umbrales
- Monitoreo continuo para nuevos patrones

### Para Clasificación Multiclase (Iris):
- ROC One-vs-Rest para múltiples clases
- Métricas macro vs micro averaging
- Visualización de fronteras de decisión

### Manejo de Datos Desbalanceados:
- Uso de `class_weight='balanced'`
- Estratificación en train-test split
- Énfasis en métricas apropiadas (F1, Recall)

## 📞 Soporte y Extensiones

Para extender este análisis:
1. Agregar validación cruzada k-fold
2. Implementar GridSearchCV para optimización de hiperparámetros
3. Comparar con otros algoritmos (SVM, Neural Networks)
4. Análisis de curvas de aprendizaje
5. Implementar técnicas de balanceo (SMOTE, undersampling)

---
**Nota:** Todos los scripts están diseñados para ser educativos y completos, cubriendo todos los aspectos requeridos para el análisis académico de modelos de Machine Learning.
