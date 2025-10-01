import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

class PerceptronClima:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0
        self.training_history = []
        
    def tanh_activation(self, x):
        """Función de activación tangente hiperbólica"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivada de la tangente hiperbólica"""
        return 1 - np.tanh(x)**2
    
    def fit(self, X, y, epochs=20):
        """Entrena el perceptrón"""
        # Inicializar pesos aleatoriamente
        self.weights = np.random.normal(0, 0.1, X.shape[1])
        self.bias = np.random.normal(0, 0.1)
        
        # Preparar log de entrenamiento
        log_content = []
        log_content.append("=== ENTRENAMIENTO PERCEPTRON PREDICCIÓN CLIMA ===")
        log_content.append("Función de activación: Tangente Hiperbólica")
        log_content.append(f"Tasa de aprendizaje: {self.learning_rate}")
        log_content.append(f"Épocas: {epochs}")
        log_content.append(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("")
        
        for epoch in range(epochs):
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.tanh_activation(linear_output)
                
                # Calcular error
                error = y[i] - prediction
                total_error += error**2
                
                # Backward pass (gradiente descendente)
                gradient = error * self.tanh_derivative(linear_output)
                
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
        self.save_to_file("resultados_clima_entrenamiento.txt", "\n".join(log_content))
    
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
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            prediction = self.tanh_activation(linear_output)
            # Convertir a clasificación binaria
            binary_prediction = 1 if prediction > 0 else 0
            predictions.append(binary_prediction)
        return np.array(predictions)
    
    def save_to_file(self, filename, content):
        """Guarda contenido en un archivo"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Resultados guardados en: {filename}")
        except Exception as e:
            print(f"Error al guardar archivo: {e}")
    
    def plot_training_history(self):
        """Grafica la historia del entrenamiento"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Historia del entrenamiento
        plt.subplot(2, 1, 1)
        plt.plot(self.training_history, 'b-', linewidth=2, label='Error de entrenamiento')
        plt.title('Historia del Entrenamiento - Predicción del Clima\n(Función Tangente Hiperbólica)', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Error Promedio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Función tangente hiperbólica
        plt.subplot(2, 1, 2)
        x = np.linspace(-3, 3, 100)
        y = np.tanh(x)
        plt.plot(x, y, 'r-', linewidth=2, label='tanh(x)')
        plt.title('Función de Activación: Tangente Hiperbólica')
        plt.xlabel('x')
        plt.ylabel('tanh(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('grafico_entrenamiento_clima.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.grid(True)
        plt.show()

# Datos de entrenamiento para predicción del clima
# Características: [temperatura, humedad, presión, viento]
# Etiquetas: 0 = No lluvia, 1 = Lluvia
def generar_datos_clima():
    # Datos sintéticos para predicción de lluvia
    X = np.array([
        [25, 60, 1013, 5],   # No lluvia
        [20, 80, 1008, 15],  # Lluvia
        [30, 40, 1020, 3],   # No lluvia
        [18, 90, 1005, 20],  # Lluvia
        [28, 50, 1015, 8],   # No lluvia
        [15, 95, 1000, 25],  # Lluvia
        [32, 35, 1025, 2],   # No lluvia
        [22, 85, 1007, 18],  # Lluvia
        [26, 55, 1018, 6],   # No lluvia
        [16, 88, 1002, 22],  # Lluvia
        [29, 45, 1022, 4],   # No lluvia
        [19, 92, 1004, 24],  # Lluvia
        [31, 38, 1028, 1],   # No lluvia
        [21, 82, 1006, 16],  # Lluvia
        [27, 58, 1016, 7],   # No lluvia
        [17, 87, 1003, 21],  # Lluvia
        [33, 42, 1024, 5],   # No lluvia
        [23, 78, 1009, 14],  # Lluvia
        [24, 62, 1014, 9],   # No lluvia
        [14, 94, 999, 26],   # Lluvia
    ])
    
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Normalizar características
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X_normalized, y

def main():
    print("=== PERCEPTRÓN PARA PREDICCIÓN DEL CLIMA ===")
    print("Función de activación: Tangente Hiperbólica\n")
    
    # Generar datos
    X, y = generar_datos_clima()
    
    # Crear y entrenar el perceptrón
    perceptron = PerceptronClima(learning_rate=0.1)
    perceptron.fit(X, y, epochs=25)
    
    # Realizar predicciones
    predictions = perceptron.predict(X)
    
    # Calcular precisión
    accuracy = np.mean(predictions == y) * 100
    print(f"\nPrecisión en datos de entrenamiento: {accuracy:.2f}%")
    
    # Preparar resultados para archivo
    results_content = []
    results_content.append("=== RESULTADOS PREDICCIÓN DEL CLIMA ===")
    results_content.append(f"Función de activación: Tangente Hiperbólica")
    results_content.append(f"Precisión: {accuracy:.2f}%")
    results_content.append(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_content.append("")
    
    # Mostrar algunos ejemplos
    print("\n=== EJEMPLOS DE PREDICCIÓN ===")
    print("Temp | Hum | Pres | Vien | Real | Pred | ¿Correcto?")
    print("-" * 55)
    
    results_content.append("=== EJEMPLOS DE PREDICCIÓN ===")
    results_content.append("Temp | Hum | Pres | Vien | Real | Pred | ¿Correcto?")
    results_content.append("-" * 55)
    
    for i in range(min(10, len(X))):
        temp_orig = X[i] * np.array([6.2, 18.5, 9.8, 8.2]) + np.array([23.0, 70.0, 1012.0, 12.0])
        real_label = "Lluvia" if y[i] == 1 else "No lluvia"
        pred_label = "Lluvia" if predictions[i] == 1 else "No lluvia"
        correct = "✓" if predictions[i] == y[i] else "✗"
        
        line = f"{temp_orig[0]:4.0f} | {temp_orig[1]:3.0f} | {temp_orig[2]:4.0f} | {temp_orig[3]:4.0f} | {real_label:8} | {pred_label:8} | {correct}"
        print(line)
        results_content.append(line)
    
    # Probar con datos nuevos
    print("\n=== PREDICCIONES EN NUEVOS DATOS ===")
    nuevos_datos = np.array([
        [13, 96, 998, 28],   # Condiciones de lluvia
        [35, 30, 1030, 1],   # Condiciones de no lluvia
        [20, 75, 1010, 12],  # Condiciones mixtas
    ])
    
    # Normalizar con los mismos parámetros
    X_original = np.array([
        [25, 60, 1013, 5], [20, 80, 1008, 15], [30, 40, 1020, 3], [18, 90, 1005, 20],
        [28, 50, 1015, 8], [15, 95, 1000, 25], [32, 35, 1025, 2], [22, 85, 1007, 18],
        [26, 55, 1018, 6], [16, 88, 1002, 22], [29, 45, 1022, 4], [19, 92, 1004, 24],
        [31, 38, 1028, 1], [21, 82, 1006, 16], [27, 58, 1016, 7], [17, 87, 1003, 21],
        [33, 42, 1024, 5], [23, 78, 1009, 14], [24, 62, 1014, 9], [14, 94, 999, 26]
    ])
    
    mean_vals = X_original.mean(axis=0)
    std_vals = X_original.std(axis=0)
    nuevos_datos_norm = (nuevos_datos - mean_vals) / std_vals
    
    nuevas_predicciones = perceptron.predict(nuevos_datos_norm)
    
    print("Temp | Hum | Pres | Vien | Predicción")
    print("-" * 40)
    
    results_content.append("")
    results_content.append("=== PREDICCIONES EN NUEVOS DATOS ===")
    results_content.append("Temp | Hum | Pres | Vien | Predicción")
    results_content.append("-" * 40)
    
    for i, pred in enumerate(nuevas_predicciones):
        pred_label = "Lluvia" if pred == 1 else "No lluvia"
        line = f"{nuevos_datos[i][0]:4.0f} | {nuevos_datos[i][1]:3.0f} | {nuevos_datos[i][2]:4.0f} | {nuevos_datos[i][3]:4.0f} | {pred_label}"
        print(line)
        results_content.append(line)
    
    results_content.append("")
    results_content.append("=== INFORMACIÓN SOBRE TANGENTE HIPERBÓLICA ===")
    results_content.append("La tangente hiperbólica es una función de activación que:")
    results_content.append("- Tiene rango (-1, 1)")
    results_content.append("- Es simétrica respecto al origen")
    results_content.append("- Tiene derivada fácil de calcular: tanh'(x) = 1 - tanh²(x)")
    results_content.append("- Centra las salidas alrededor de 0")
    results_content.append("- Es útil para problemas de clasificación binaria")
    
    # Guardar resultados completos
    perceptron.save_to_file("resultados_clima_completos.txt", "\n".join(results_content))
    
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
