import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

class PerceptronFraude:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0
        self.training_history = []
        
    def sigmoid_activation(self, x):
        """Función de activación sigmoidal"""
        # Evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivada de la función sigmoidal"""
        sig = self.sigmoid_activation(x)
        return sig * (1 - sig)
    
    def fit(self, X, y, epochs=30):
        """Entrena el perceptrón"""
        # Inicializar pesos aleatoriamente
        self.weights = np.random.normal(0, 0.1, X.shape[1])
        self.bias = np.random.normal(0, 0.1)
        
        # Preparar log de entrenamiento
        log_content = []
        log_content.append("=== ENTRENAMIENTO PERCEPTRON DETECCIÓN FRAUDE ===")
        log_content.append("Función de activación: Sigmoidal")
        log_content.append(f"Tasa de aprendizaje: {self.learning_rate}")
        log_content.append(f"Épocas: {epochs}")
        log_content.append(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("")
        
        for epoch in range(epochs):
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.sigmoid_activation(linear_output)
                
                # Calcular error
                error = y[i] - prediction
                total_error += error**2
                
                # Backward pass (gradiente descendente)
                gradient = error * self.sigmoid_derivative(linear_output)
                
                # Actualizar pesos y bias
                self.weights += self.learning_rate * gradient * X[i]
                self.bias += self.learning_rate * gradient
            
            avg_error = total_error / len(X)
            self.training_history.append(avg_error)
            
            if epoch % 5 == 0:
                message = f"Época {epoch}, Error promedio: {avg_error:.4f}"
                print(message)
                log_content.append(message)
        
        # Guardar pesos finales
        log_content.append("")
        log_content.append("=== PESOS FINALES ===")
        for i, weight in enumerate(self.weights):
            log_content.append(f"Peso {i}: {weight:.4f}")
        log_content.append(f"Bias: {self.bias:.4f}")
        
        # Guardar log en archivo
        self.save_to_file("resultados_fraude_entrenamiento.txt", "\n".join(log_content))
    
    def save_to_file(self, filename, content):
        """Guarda contenido en un archivo"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Resultados guardados en: {filename}")
        except Exception as e:
            print(f"Error al guardar archivo: {e}")
    
    def predict(self, X):
        """Realiza predicciones"""
        predictions = []
        probabilities = []
        
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            probability = self.sigmoid_activation(linear_output)
            prediction = 1 if probability > 0.5 else 0
            
            predictions.append(prediction)
            probabilities.append(probability)
            
        return np.array(predictions), np.array(probabilities)
    
    def plot_training_history(self):
        """Grafica la historia del entrenamiento"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Historia del entrenamiento
        plt.subplot(2, 1, 1)
        plt.plot(self.training_history, 'b-', linewidth=2, label='Error de entrenamiento')
        plt.title('Historia del Entrenamiento - Detección de Fraude\n(Función Sigmoidal)', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Error Promedio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Función sigmoidal
        plt.subplot(2, 1, 2)
        x = np.linspace(-6, 6, 100)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y, 'r-', linewidth=2, label='sigmoid(x)')
        plt.title('Función de Activación: Sigmoidal')
        plt.xlabel('x')
        plt.ylabel('sigmoid(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Umbral 0.5')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('grafico_entrenamiento_fraude.png', dpi=300, bbox_inches='tight')
        plt.show()

# Datos de entrenamiento para detección de fraude
# Características: [monto_transaccion, hora_del_dia, ubicacion_inusual, frecuencia_transacciones, edad_cuenta]
# Etiquetas: 0 = Legítima, 1 = Fraude
def generar_datos_fraude():
    # Datos sintéticos para detección de fraude
    X = np.array([
        [50, 14, 0, 3, 365],     # Legítima
        [5000, 3, 1, 15, 30],    # Fraude
        [120, 10, 0, 5, 180],    # Legítima
        [8000, 2, 1, 20, 10],    # Fraude
        [75, 16, 0, 2, 450],     # Legítima
        [3500, 1, 1, 18, 5],     # Fraude
        [200, 12, 0, 4, 280],    # Legítima
        [7500, 4, 1, 25, 15],    # Fraude
        [90, 15, 0, 3, 600],     # Legítima
        [4200, 0, 1, 22, 8],     # Fraude
        [150, 11, 0, 6, 320],    # Legítima
        [6800, 23, 1, 17, 12],   # Fraude
        [80, 13, 0, 4, 500],     # Legítima
        [9200, 2, 1, 30, 7],     # Fraude
        [110, 17, 0, 5, 400],    # Legítima
        [5500, 1, 1, 19, 20],    # Fraude
        [65, 9, 0, 2, 720],      # Legítima
        [7800, 3, 1, 28, 6],     # Fraude
        [140, 14, 0, 7, 250],    # Legítima
        [4800, 0, 1, 16, 14],    # Fraude
        [95, 16, 0, 3, 580],     # Legítima
        [6200, 4, 1, 24, 9],     # Fraude
        [180, 10, 0, 5, 330],    # Legítima
        [8500, 23, 1, 21, 11],   # Fraude
        [70, 12, 0, 4, 480],     # Legítima
        [3800, 2, 1, 26, 18],    # Fraude
        [125, 15, 0, 6, 290],    # Legítima
        [7200, 1, 1, 23, 13],    # Fraude
        [100, 11, 0, 3, 640],    # Legítima
        [5800, 4, 1, 27, 16],    # Fraude
    ])
    
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                  0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Normalizar características
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X_normalized, y

def main():
    print("=== PERCEPTRÓN PARA DETECCIÓN DE FRAUDE ===")
    print("Función de activación: Sigmoidal\n")
    
    # Generar datos
    X, y = generar_datos_fraude()
    
    # Crear y entrenar el perceptrón
    perceptron = PerceptronFraude(learning_rate=0.5)
    perceptron.fit(X, y, epochs=35)
    
    # Realizar predicciones
    predictions, probabilities = perceptron.predict(X)
    
    # Calcular precisión
    accuracy = np.mean(predictions == y) * 100
    print(f"\nPrecisión en datos de entrenamiento: {accuracy:.2f}%")
    
    # Preparar resultados para archivo
    results_content = []
    results_content.append("=== RESULTADOS DETECCIÓN DE FRAUDE ===")
    results_content.append(f"Función de activación: Sigmoidal")
    results_content.append(f"Precisión: {accuracy:.2f}%")
    results_content.append(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_content.append("")
    
    # Mostrar algunos ejemplos
    print("\n=== EJEMPLOS DE PREDICCIÓN ===")
    print("Mont | Hora | Ubic | Freq | Edad | Real      | Pred      | Prob  | ¿Correcto?")
    print("-" * 80)
    
    results_content.append("=== EJEMPLOS DE PREDICCIÓN ===")
    results_content.append("Mont | Hora | Ubic | Freq | Edad | Real      | Pred      | Prob  | ¿Correcto?")
    results_content.append("-" * 80)
    
    # Datos originales para mostrar
    X_original = np.array([
        [50, 14, 0, 3, 365], [5000, 3, 1, 15, 30], [120, 10, 0, 5, 180], [8000, 2, 1, 20, 10],
        [75, 16, 0, 2, 450], [3500, 1, 1, 18, 5], [200, 12, 0, 4, 280], [7500, 4, 1, 25, 15],
        [90, 15, 0, 3, 600], [4200, 0, 1, 22, 8], [150, 11, 0, 6, 320], [6800, 23, 1, 17, 12],
        [80, 13, 0, 4, 500], [9200, 2, 1, 30, 7], [110, 17, 0, 5, 400], [5500, 1, 1, 19, 20]
    ])
    
    for i in range(min(16, len(X))):
        real_label = "Fraude   " if y[i] == 1 else "Legítima "
        pred_label = "Fraude   " if predictions[i] == 1 else "Legítima "
        correct = "✓" if predictions[i] == y[i] else "✗"
        
        line = f"{X_original[i][0]:4.0f} | {X_original[i][1]:4.0f} | {X_original[i][2]:4.0f} | {X_original[i][3]:4.0f} | {X_original[i][4]:4.0f} | {real_label} | {pred_label} | {probabilities[i]:.3f} | {correct}"
        print(line)
        results_content.append(line)
    
    # Probar con datos nuevos
    print("\n=== PREDICCIONES EN NUEVOS DATOS ===")
    nuevos_datos = np.array([
        [10000, 2, 1, 35, 5],    # Muy sospechoso (monto alto, hora extraña, ubicación inusual, etc.)
        [45, 12, 0, 2, 800],     # Muy normal (monto bajo, hora normal, ubicación normal, etc.)
        [2000, 15, 1, 8, 60],    # Moderadamente sospechoso
        [85, 20, 0, 4, 200],     # Normal pero con hora un poco tardía
    ])
    
    # Normalizar con los mismos parámetros del entrenamiento
    X_train_original = np.array([
        [50, 14, 0, 3, 365], [5000, 3, 1, 15, 30], [120, 10, 0, 5, 180], [8000, 2, 1, 20, 10],
        [75, 16, 0, 2, 450], [3500, 1, 1, 18, 5], [200, 12, 0, 4, 280], [7500, 4, 1, 25, 15],
        [90, 15, 0, 3, 600], [4200, 0, 1, 22, 8], [150, 11, 0, 6, 320], [6800, 23, 1, 17, 12],
        [80, 13, 0, 4, 500], [9200, 2, 1, 30, 7], [110, 17, 0, 5, 400], [5500, 1, 1, 19, 20],
        [65, 9, 0, 2, 720], [7800, 3, 1, 28, 6], [140, 14, 0, 7, 250], [4800, 0, 1, 16, 14],
        [95, 16, 0, 3, 580], [6200, 4, 1, 24, 9], [180, 10, 0, 5, 330], [8500, 23, 1, 21, 11],
        [70, 12, 0, 4, 480], [3800, 2, 1, 26, 18], [125, 15, 0, 6, 290], [7200, 1, 1, 23, 13],
        [100, 11, 0, 3, 640], [5800, 4, 1, 27, 16]
    ])
    
    mean_vals = X_train_original.mean(axis=0)
    std_vals = X_train_original.std(axis=0)
    nuevos_datos_norm = (nuevos_datos - mean_vals) / std_vals
    
    nuevas_predicciones, nuevas_probabilidades = perceptron.predict(nuevos_datos_norm)
    
    print("Mont | Hora | Ubic | Freq | Edad | Predicción | Probabilidad | Nivel de Riesgo")
    print("-" * 75)
    
    results_content.append("")
    results_content.append("=== PREDICCIONES EN NUEVOS DATOS ===")
    results_content.append("Mont | Hora | Ubic | Freq | Edad | Predicción | Probabilidad | Nivel de Riesgo")
    results_content.append("-" * 75)
    
    for i, (pred, prob) in enumerate(zip(nuevas_predicciones, nuevas_probabilidades)):
        pred_label = "Fraude   " if pred == 1 else "Legítima "
        
        if prob > 0.8:
            riesgo = "Muy Alto"
        elif prob > 0.6:
            riesgo = "Alto"
        elif prob > 0.4:
            riesgo = "Medio"
        elif prob > 0.2:
            riesgo = "Bajo"
        else:
            riesgo = "Muy Bajo"
        
        line = f"{nuevos_datos[i][0]:4.0f} | {nuevos_datos[i][1]:4.0f} | {nuevos_datos[i][2]:4.0f} | {nuevos_datos[i][3]:4.0f} | {nuevos_datos[i][4]:4.0f} | {pred_label} | {prob:11.3f} | {riesgo}"
        print(line)
        results_content.append(line)
    
    # Mostrar estadísticas del modelo
    print(f"\n=== ESTADÍSTICAS DEL MODELO ===")
    true_positives = np.sum((predictions == 1) & (y == 1))
    false_positives = np.sum((predictions == 1) & (y == 0))
    true_negatives = np.sum((predictions == 0) & (y == 0))
    false_negatives = np.sum((predictions == 0) & (y == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precisión: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    print(f"Verdaderos Positivos: {true_positives}")
    print(f"Falsos Positivos: {false_positives}")
    print(f"Verdaderos Negativos: {true_negatives}")
    print(f"Falsos Negativos: {false_negatives}")
    
    results_content.append("")
    results_content.append("=== ESTADÍSTICAS DEL MODELO ===")
    results_content.append(f"Precisión: {precision:.3f}")
    results_content.append(f"Recall: {recall:.3f}")
    results_content.append(f"F1-Score: {f1_score:.3f}")
    results_content.append(f"Verdaderos Positivos: {true_positives}")
    results_content.append(f"Falsos Positivos: {false_positives}")
    results_content.append(f"Verdaderos Negativos: {true_negatives}")
    results_content.append(f"Falsos Negativos: {false_negatives}")
    
    results_content.append("")
    results_content.append("=== INFORMACIÓN SOBRE FUNCIÓN SIGMOIDAL ===")
    results_content.append("La función sigmoidal es ideal para clasificación binaria porque:")
    results_content.append("- Tiene rango (0, 1), interpretable como probabilidad")
    results_content.append("- Es suave y diferenciable en todo su dominio")
    results_content.append("- Fórmula: σ(x) = 1/(1 + e^(-x))")
    results_content.append("- Derivada: σ'(x) = σ(x) * (1 - σ(x))")
    results_content.append("- Umbral natural en 0.5 para clasificación")
    
    # Guardar resultados completos
    perceptron.save_to_file("resultados_fraude_completos.txt", "\n".join(results_content))
    
    # Graficar historia del entrenamiento
    try:
        perceptron.plot_training_history()
    except Exception as e:
        print(f"\nNo se pudo mostrar el gráfico: {e}")
        print("\nHistoria del entrenamiento:")
        for i, error in enumerate(perceptron.training_history):
            if i % 5 == 0:
                print(f"Época {i}: Error = {error:.4f}")

if __name__ == "__main__":
    main()
