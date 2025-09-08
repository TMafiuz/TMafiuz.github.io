import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
print("ANÁLISIS DE DETECCIÓN DE FRAUDE")
print("Árboles de Decisión vs Random Forest")
print("=" * 80)

# 2. CARGA Y EXPLORACIÓN DE DATOS
print("\n2. CARGA Y EXPLORACIÓN DE DATOS")
print("-" * 40)

# Cargar los diferentes archivos de datos
print("Cargando archivos de datos...")

# Cargar datos de transacciones
transaction_records = pd.read_csv(r'Taller_3\Fraude\Data\Transaction Data\transaction_records.csv')
fraud_indicators = pd.read_csv(r'Taller_3\Fraude\Data\Fraudulent Patterns\fraud_indicators.csv')
customer_data = pd.read_csv(r'Taller_3\Fraude\Data\Customer Profiles\customer_data.csv')
amount_data = pd.read_csv(r'Taller_3\Fraude\Data\Transaction Amounts\amount_data.csv')

print("✓ Archivos cargados exitosamente")

# Combinar los datos
print("\nCombinando datasets...")
# Unir transacciones con indicadores de fraude
df_base = pd.merge(transaction_records, fraud_indicators, on='TransactionID', how='inner')

# Unir con datos de clientes
df_base = pd.merge(df_base, customer_data, on='CustomerID', how='left')

# Unir con datos de montos (si es diferente de Amount en transaction_records)
if 'TransactionAmount' in amount_data.columns:
    df_base = pd.merge(df_base, amount_data, on='TransactionID', how='left')
    # Si hay diferencias entre Amount y TransactionAmount, usar TransactionAmount
    df_base['Amount'] = df_base['TransactionAmount'].fillna(df_base['Amount'])
    df_base = df_base.drop('TransactionAmount', axis=1)

print(f"✓ Dataset combinado creado")
print(f"Dimensiones del dataset: {df_base.shape}")
print(f"Variables disponibles: {list(df_base.columns)}")

# Verificar datos
print(f"\nPrimeras 5 filas del dataset combinado:")
print(df_base.head())

print(f"\nInformación sobre valores faltantes:")
print(df_base.isnull().sum())

print(f"\nDistribución de fraude (variable objetivo):")
fraud_counts = df_base['FraudIndicator'].value_counts()
print(fraud_counts)
print(f"Tasa de fraude: {df_base['FraudIndicator'].mean():.2%}")

# Estadísticas de las transacciones
print(f"\nEstadísticas de transacciones:")
print(f"Monto promedio: ${df_base['Amount'].mean():.2f}")
print(f"Monto mediano: ${df_base['Amount'].median():.2f}")
print(f"Monto mínimo: ${df_base['Amount'].min():.2f}")
print(f"Monto máximo: ${df_base['Amount'].max():.2f}")

print(f"\nEdad promedio de clientes: {df_base['Age'].mean():.1f} años")

# 3. PREPROCESAMIENTO DE DATOS
print("\n3. PREPROCESAMIENTO DE DATOS")
print("-" * 40)

# Crear una copia para el preprocesamiento
data = df_base.copy()

print("Pasos de preprocesamiento:")
print("1. Manejo de valores faltantes")
print("2. Creación de características derivadas")
print("3. Codificación de variables categóricas")

# Manejo de valores faltantes
data['Age'].fillna(data['Age'].median(), inplace=True)
print("✓ Valores faltantes de Age imputados con la mediana")

# Crear características derivadas para mejorar la detección
print("\nCreando características derivadas:")

# Características basadas en el monto de la transacción
data['AmountZScore'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()
data['IsHighAmount'] = (data['Amount'] > data['Amount'].quantile(0.95)).astype(int)
data['IsLowAmount'] = (data['Amount'] < data['Amount'].quantile(0.05)).astype(int)

# Características basadas en la edad del cliente
data['IsYoungCustomer'] = (data['Age'] < 25).astype(int)
data['IsOldCustomer'] = (data['Age'] > 60).astype(int)

# Característica de riesgo combinada
data['RiskScore'] = data['IsHighAmount'] + data['IsYoungCustomer'] + data['IsOldCustomer']

print("✓ Características derivadas creadas:")
print("  - AmountZScore: Puntuación Z del monto")
print("  - IsHighAmount: Indicador de monto alto")
print("  - IsLowAmount: Indicador de monto bajo")
print("  - IsYoungCustomer: Cliente joven (<25 años)")
print("  - IsOldCustomer: Cliente mayor (>60 años)")
print("  - RiskScore: Puntuación de riesgo combinada")

# Seleccionar características finales para el modelo
feature_columns = [
    'Amount', 'Age', 'AmountZScore', 'IsHighAmount', 'IsLowAmount', 
    'IsYoungCustomer', 'IsOldCustomer', 'RiskScore'
]

X = data[feature_columns]
y = data['FraudIndicator']

print(f"\nCaracterísticas finales: {feature_columns}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# División de datos con estratificación para manejar el desbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisión de datos:")
print(f"- Entrenamiento: {X_train.shape[0]} muestras")
print(f"  * Fraudes: {y_train.sum()} ({y_train.mean():.2%})")
print(f"  * Legítimas: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
print(f"- Prueba: {X_test.shape[0]} muestras") 
print(f"  * Fraudes: {y_test.sum()} ({y_test.mean():.2%})")
print(f"  * Legítimas: {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")

# 4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS
print("\n4. CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS")
print("-" * 40)

# 4.1 Árbol de Decisión
print("\n4.1 ÁRBOL DE DECISIÓN - OPTIMIZADO PARA DETECCIÓN DE FRAUDE")
print("=" * 65)
print("DESAFÍO: Detectar fraudes en un dataset EXTREMADAMENTE DESBALANCEADO")
print("• Solo 4.5% de transacciones son fraudulentas")
print("• El modelo debe ser muy sensible para capturar patrones raros")
print("• Balance crítico entre detectar fraudes y evitar falsas alarmas")
print()
print("CONFIGURACIÓN ESPECIALIZADA PARA FRAUDE:")
print("┌─ max_depth=8:")
print("│  • Mayor profundidad que otros casos (Titanic=5, Iris=4)")
print("│  • Permite capturar patrones complejos de fraude")
print("│  • Los fraudes pueden tener reglas de detección más elaboradas")
print("│  • Riesgo controlado de overfitting por class_weight='balanced'")
print("│")
print("┌─ min_samples_split=10:")
print("│  • Reducido vs Titanic (20) para mayor sensibilidad")
print("│  • Permite divisiones con menos muestras (importante para clase minoritaria)")
print("│  • Facilita la detección de patrones de fraude raros")
print("│")
print("┌─ min_samples_leaf=5:")
print("│  • Hojas más pequeñas para capturar casos específicos de fraude")
print("│  • Balance entre especificidad y generalización")
print("│  • Permite reglas de detección muy precisas")
print("│")
print("┌─ class_weight='balanced': ¡CRÍTICO PARA FRAUDE!")
print("│  • Automáticamente ajusta pesos: Legítimo=0.52, Fraude=11.1")
print("│  • Penaliza MÁS los errores en la clase minoritaria (fraudes)")
print("│  • Fuerza al modelo a prestar más atención a los fraudes")
print("│  • Sin esto, el modelo ignoraría completamente los fraudes")
print("│")
print("└─ random_state=42: Reproducibilidad de experimentos")
print()
print("ESTRATEGIA DE DETECCIÓN:")
print("• Priorizar RECALL sobre Precision (mejor detectar fraudes extra)")
print("• Un fraude no detectado cuesta $100, una falsa alarma $1")
print("• El árbol creará reglas muy específicas para patrones fraudulentos")
print()

dt_classifier = DecisionTreeClassifier(
    max_depth=8,           
    min_samples_split=10,  
    min_samples_leaf=5,    
    class_weight='balanced',
    random_state=42
)

print("Entrenando Árbol de Decisión especializado en fraude...")
dt_classifier.fit(X_train, y_train)

# Calcular pesos reales aplicados cuando class_weight='balanced'
# Fórmula: n_samples / (n_classes * np.bincount(y))
weight_0 = len(y_train) / (2 * (y_train == 0).sum())  # Peso para clase legítima
weight_1 = len(y_train) / (2 * (y_train == 1).sum())  # Peso para clase fraude

print("✓ Árbol de Decisión entrenado exitosamente")
print(f"  • Profundidad máxima configurada: {dt_classifier.max_depth}")
print(f"  • Profundidad real alcanzada: {dt_classifier.get_depth()}")
print(f"  • Número de hojas: {dt_classifier.get_n_leaves()}")
print(f"  • Peso clase legítima (0): {weight_0:.3f}")
print(f"  • Peso clase fraude (1): {weight_1:.1f} ← ¡{weight_1/weight_0:.1f}x más importante!")
print(f"  • Balance de clases: {'✓ ACTIVADO' if dt_classifier.class_weight == 'balanced' else '✗ Desactivado'}")

# 4.2 Random Forest
print("\n4.2 RANDOM FOREST - ENSEMBLE PARA DETECCIÓN DE FRAUDE")
print("=" * 65)
print("TEORÍA DEL ENSEMBLE PARA FRAUDE:")
print("• Combina 100 'detectores de fraude' independientes")
print("• Cada árbol ve una muestra diferente de transacciones (bootstrap)")
print("• Algunos árboles se especializarán en diferentes tipos de fraude")
print("• El voto mayoritario reduce falsos positivos")
print()
print("¿POR QUÉ ENSEMBLE EN FRAUDE?")
print("┌─ Diversidad de Patrones:")
print("│  • Fraude por monto alto vs fraude por cliente joven")
print("│  • Diferentes árboles pueden especializarse en cada patrón")
print("│  • Captura múltiples 'firmas' de comportamiento fraudulento")
print("│")
print("┌─ Robustez ante Outliers:")
print("│  • Los fraudes son inherentemente 'outliers'")
print("│  • Un árbol individual podría ser engañado por casos raros")
print("│  • El ensemble es más estable ante variaciones extremas")
print("│")
print("┌─ Reducción de Varianza:")
print("│  • Importante cuando hay pocos ejemplos de fraude")
print("│  • Cada árbol entrena con ~32 fraudes (bootstrap)")
print("│  • El promedio de 100 árboles es más confiable")
print("│")
print("└─ Estimación de Confianza:")
print("   • % de árboles que votan 'fraude' = nivel de sospecha")
print("   • Útil para establecer umbrales de alerta")
print()
print("CONFIGURACIÓN DEL ENSEMBLE:")
print("┌─ n_estimators=100:")
print("│  • 100 detectores independientes")
print("│  • Cada uno entrena con ~640 transacciones (bootstrap)")
print("│  • Balance entre diversidad y eficiencia computacional")
print("│")
print("┌─ max_depth=8:")
print("│  • Misma profundidad que árbol individual")
print("│  • Cada árbol puede crear reglas complejas")
print("│  • La diversidad viene del bootstrap, no de árboles simples")
print("│")
print("┌─ class_weight='balanced':")
print("│  • ¡MUY IMPORTANTE! Aplicado a cada árbol del ensemble")
print("│  • Sin esto, Random Forest ignoraría completamente los fraudes")
print("│  • Cada árbol individual está sesgado hacia detección")
print("│")
print("└─ n_jobs=-1: Paralelización (100 árboles toman tiempo)")
print()

rf_classifier = RandomForestClassifier(
    n_estimators=100,      
    max_depth=8,           
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1              
)

print("Entrenando Random Forest (100 detectores de fraude)...")
rf_classifier.fit(X_train, y_train)
print("✓ Random Forest entrenado exitosamente")
print(f"  • Número de detectores (árboles): {rf_classifier.n_estimators}")
print(f"  • Profundidad máxima por detector: {rf_classifier.max_depth}")
print(f"  • Transacciones por detector: ~{int(0.632 * len(X_train))}")
print(f"  • Fraudes esperados por detector: ~{int(0.632 * (y_train == 1).sum())}")
print(f"  • Características evaluadas por split: √8 ≈ 3")
print(f"  • Balance de clases: {'✓ ACTIVADO en cada árbol' if rf_classifier.class_weight == 'balanced' else '✗ Desactivado'}")
print(f"  • Procesamiento paralelo: ✓ Todos los cores")

print("\n⚠️  DESAFÍO ESPECÍFICO DEL ENSEMBLE EN FRAUDE:")
print("• Random Forest puede ser 'demasiado conservador' con clases desbalanceadas")
print("• Cada árbol ve pocos fraudes (bootstrap sampling)")
print("• El voto mayoritario puede diluir señales de fraude débiles")
print("• class_weight='balanced' es ESENCIAL para compensar este efecto")
print("• Alternativamente se podría usar técnicas de oversampling (SMOTE)")

# 5. EVALUACIÓN Y COMPARACIÓN DE MODELOS
print("\n5. EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("-" * 40)

# Predicciones
dt_pred = dt_classifier.predict(X_test)
dt_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

rf_pred = rf_classifier.predict(X_test)
rf_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Función para calcular métricas de fraude
def calcular_metricas_fraude(y_true, y_pred, y_pred_proba, modelo_nombre):
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
    
    # En detección de fraude, el Recall es CRÍTICO
    print(f"\n⚠️  ANÁLISIS CRÍTICO PARA FRAUDE:")
    print(f"   Recall (Sensibilidad): {recall:.4f}")
    print(f"   → Porcentaje de fraudes detectados: {recall:.1%}")
    print(f"   Precision: {precision:.4f}")
    print(f"   → Porcentaje de alertas que son fraude real: {precision:.1%}")
    
    # Matriz de confusión con interpretación de fraude
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nMatriz de Confusión:")
    print(f"                 Predicho")
    print(f"              Legítimo  Fraude")
    print(f"Real Legítimo    {tn:6d}   {fp:6d}")
    print(f"     Fraude      {fn:6d}   {tp:6d}")
    
    print(f"\nInterpretación de errores:")
    print(f"• Falsos Positivos (FP): {fp} - Transacciones legítimas marcadas como fraude")
    print(f"• Falsos Negativos (FN): {fn} - ¡FRAUDES NO DETECTADOS! (MUY CRÍTICO)")
    print(f"• Verdaderos Positivos (TP): {tp} - Fraudes correctamente detectados")
    print(f"• Verdaderos Negativos (TN): {tn} - Transacciones legítimas correctas")
    
    # Especificidad (importante para reducir falsos positivos)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"• Especificidad: {specificity:.4f} - Porcentaje de legítimas correctamente clasificadas")
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"• AUC-ROC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 
        'f1': f1, 'auc': roc_auc, 'specificity': specificity,
        'fpr': fpr, 'tpr': tpr, 'cm': cm, 
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# Calcular métricas para ambos modelos
dt_metrics = calcular_metricas_fraude(y_test, dt_pred, dt_pred_proba, "Árbol de Decisión")
rf_metrics = calcular_metricas_fraude(y_test, rf_pred, rf_pred_proba, "Random Forest")

# 6. ANÁLISIS COMPARATIVO
print("\n6. ANÁLISIS COMPARATIVO")
print("-" * 40)

# Comparación de métricas
metricas_comparacion = pd.DataFrame({
    'Árbol de Decisión': [
        dt_metrics['accuracy'], dt_metrics['precision'], 
        dt_metrics['recall'], dt_metrics['f1'], dt_metrics['auc']
    ],
    'Random Forest': [
        rf_metrics['accuracy'], rf_metrics['precision'], 
        rf_metrics['recall'], rf_metrics['f1'], rf_metrics['auc']
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])

print("Comparación de Métricas:")
print(metricas_comparacion.round(4))

# Para fraude, el Recall es más importante que la Precision
print(f"\n🎯 ANÁLISIS ESPECÍFICO PARA DETECCIÓN DE FRAUDE:")
print(f"{'Métrica':<15} {'Árbol':<10} {'Random Forest':<15} {'Mejor':<15}")
print("-" * 55)
print(f"{'Recall':<15} {dt_metrics['recall']:<10.4f} {rf_metrics['recall']:<15.4f} "
      f"{'Random Forest' if rf_metrics['recall'] > dt_metrics['recall'] else 'Árbol':<15}")
print(f"{'Precision':<15} {dt_metrics['precision']:<10.4f} {rf_metrics['precision']:<15.4f} "
      f"{'Random Forest' if rf_metrics['precision'] > dt_metrics['precision'] else 'Árbol':<15}")
print(f"{'F1-Score':<15} {dt_metrics['f1']:<10.4f} {rf_metrics['f1']:<15.4f} "
      f"{'Random Forest' if rf_metrics['f1'] > dt_metrics['f1'] else 'Árbol':<15}")

# Costo de errores en fraude
costo_fn = 100  # Costo de no detectar un fraude (muy alto)
costo_fp = 1    # Costo de una falsa alarma (bajo)

dt_costo = (dt_metrics['fn'] * costo_fn) + (dt_metrics['fp'] * costo_fp)
rf_costo = (rf_metrics['fn'] * costo_fn) + (rf_metrics['fp'] * costo_fp)

print(f"\n💰 ANÁLISIS DE COSTO-BENEFICIO:")
print(f"Asumiendo: Costo de fraude no detectado = ${costo_fn}, Costo de falsa alarma = ${costo_fp}")
print(f"Costo total Árbol de Decisión: ${dt_costo}")
print(f"Costo total Random Forest: ${rf_costo}")
mejor_costo = 'Random Forest' if rf_costo < dt_costo else 'Árbol de Decisión'
print(f"Mejor modelo por costo: {mejor_costo}")

# 7. VISUALIZACIONES
print("\n7. GENERANDO VISUALIZACIONES")
print("-" * 40)

# Configurar subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis Detección de Fraude: Árboles de Decisión vs Random Forest', fontsize=16)

# 7.1 Distribución de montos por fraude
axes[0,0].hist(data[data['FraudIndicator']==0]['Amount'], bins=30, alpha=0.7, 
              label='Transacciones Legítimas', color='green')
axes[0,0].hist(data[data['FraudIndicator']==1]['Amount'], bins=30, alpha=0.7, 
              label='Transacciones Fraudulentas', color='red')
axes[0,0].set_xlabel('Monto de Transacción')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].set_title('Distribución de Montos por Tipo')
axes[0,0].legend()

# 7.2 Distribución de edades por fraude
axes[0,1].hist(data[data['FraudIndicator']==0]['Age'], bins=20, alpha=0.7, 
              label='Clientes Legítimos', color='green')
axes[0,1].hist(data[data['FraudIndicator']==1]['Age'], bins=20, alpha=0.7, 
              label='Clientes con Fraude', color='red')
axes[0,1].set_xlabel('Edad del Cliente')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].set_title('Distribución de Edades por Tipo')
axes[0,1].legend()

# 7.3 Matriz de Confusión - Árbol de Decisión
sns.heatmap(dt_metrics['cm'], annot=True, fmt='d', cmap='Blues',
           xticklabels=['Legítimo', 'Fraude'],
           yticklabels=['Legítimo', 'Fraude'], ax=axes[0,2])
axes[0,2].set_title('Matriz de Confusión - Árbol de Decisión')
axes[0,2].set_ylabel('Valores Reales')
axes[0,2].set_xlabel('Valores Predichos')

# 7.4 Matriz de Confusión - Random Forest
sns.heatmap(rf_metrics['cm'], annot=True, fmt='d', cmap='Greens',
           xticklabels=['Legítimo', 'Fraude'],
           yticklabels=['Legítimo', 'Fraude'], ax=axes[1,0])
axes[1,0].set_title('Matriz de Confusión - Random Forest')
axes[1,0].set_ylabel('Valores Reales')
axes[1,0].set_xlabel('Valores Predichos')

# 7.5 Comparación de Métricas
metricas_comparacion.plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Comparación de Métricas')
axes[1,1].set_ylabel('Puntuación')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].legend()

# 7.6 Curvas ROC
axes[1,2].plot(dt_metrics['fpr'], dt_metrics['tpr'], 
              label=f'Árbol de Decisión (AUC = {dt_metrics["auc"]:.3f})', linewidth=2)
axes[1,2].plot(rf_metrics['fpr'], rf_metrics['tpr'], 
              label=f'Random Forest (AUC = {rf_metrics["auc"]:.3f})', linewidth=2)
axes[1,2].plot([0, 1], [0, 1], 'k--', label='Línea Base')
axes[1,2].set_xlabel('Tasa de Falsos Positivos')
axes[1,2].set_ylabel('Tasa de Verdaderos Positivos')
axes[1,2].set_title('Curvas ROC')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. IMPORTANCIA DE CARACTERÍSTICAS
print("\n8. IMPORTANCIA DE CARACTERÍSTICAS")
print("-" * 40)

importancias_rf = rf_classifier.feature_importances_
importancias_dt = dt_classifier.feature_importances_
indices = np.argsort(importancias_rf)[::-1]

print("Ranking de importancia de características:")
print(f"{'Rank':<5} {'Característica':<20} {'Random Forest':<15} {'Árbol Decisión':<15}")
print("-" * 60)
for i in range(len(feature_columns)):
    idx = indices[i]
    print(f"{i+1:<5} {feature_columns[idx]:<20} {importancias_rf[idx]:<15.4f} {importancias_dt[idx]:<15.4f}")

# Visualización de importancia
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(feature_columns))
width = 0.35

plt.bar(x_pos - width/2, importancias_rf[indices], width, label='Random Forest', alpha=0.8)
plt.bar(x_pos + width/2, importancias_dt[indices], width, label='Árbol de Decisión', alpha=0.8)

plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Comparación de Importancia de Características')
plt.xticks(x_pos, [feature_columns[i] for i in indices], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('fraud_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. ANÁLISIS DE UMBRAL ÓPTIMO
print("\n9. ANÁLISIS DE UMBRAL ÓPTIMO")
print("-" * 40)

# Para Random Forest (mejor modelo), analizar diferentes umbrales
umbrales = np.linspace(0.1, 0.9, 9)
resultados_umbral = []

print("Análisis de umbral para Random Forest:")
print(f"{'Umbral':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Fraudes Detectados':<17}")
print("-" * 55)

for umbral in umbrales:
    pred_umbral = (rf_pred_proba >= umbral).astype(int)
    precision_u = precision_score(y_test, pred_umbral)
    recall_u = recall_score(y_test, pred_umbral)
    f1_u = f1_score(y_test, pred_umbral)
    fraudes_detectados = recall_u * y_test.sum()
    
    resultados_umbral.append([umbral, precision_u, recall_u, f1_u, fraudes_detectados])
    print(f"{umbral:<8.1f} {precision_u:<10.3f} {recall_u:<8.3f} {f1_u:<8.3f} {fraudes_detectados:<17.1f}")

# Encontrar umbral óptimo (balance entre precision y recall)
resultados_df = pd.DataFrame(resultados_umbral, 
                           columns=['Umbral', 'Precision', 'Recall', 'F1', 'FraudesDetectados'])
umbral_optimo = resultados_df.loc[resultados_df['F1'].idxmax(), 'Umbral']
print(f"\nUmbral óptimo basado en F1-Score: {umbral_optimo:.1f}")

# 10. CONCLUSIONES
print("\n10. CONCLUSIONES")
print("-" * 40)
print("🔍 HALLAZGOS PRINCIPALES:")

# Comparar rendimiento general
if rf_metrics['recall'] > dt_metrics['recall']:
    print("• Random Forest supera al Árbol de Decisión en detección de fraudes (mayor Recall)")
else:
    print("• Árbol de Decisión supera a Random Forest en detección de fraudes (mayor Recall)")

print(f"• Tasa de fraudes detectados:")
print(f"  - Árbol de Decisión: {dt_metrics['recall']:.1%}")
print(f"  - Random Forest: {rf_metrics['recall']:.1%}")

print(f"• Precisión en alertas:")
print(f"  - Árbol de Decisión: {dt_metrics['precision']:.1%}")
print(f"  - Random Forest: {rf_metrics['precision']:.1%}")

# Características más importantes
top_feature = feature_columns[np.argmax(importancias_rf)]
print(f"• Característica más importante: {top_feature}")

# Resumen ejecutivo
print(f"\n📊 RESUMEN EJECUTIVO:")
mejor_modelo_fraude = 'Random Forest' if rf_metrics['recall'] > dt_metrics['recall'] else 'Árbol de Decisión'
print(f"• Mejor modelo para detección de fraude: {mejor_modelo_fraude}")
print(f"• Tasa de detección de fraudes: {max(rf_metrics['recall'], dt_metrics['recall']):.1%}")
print(f"• Falsos negativos (fraudes no detectados): {min(rf_metrics['fn'], dt_metrics['fn'])} de {y_test.sum()}")
print(f"• El modelo puede identificar efectivamente patrones fraudulentos")
