using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;

public class PerceptronRiesgoAcademico
{
    private double[] weights;
    private double bias;
    private double learningRate;
    private List<double> trainingHistory;
    
    public PerceptronRiesgoAcademico(double learningRate = 0.1)
    {
        this.learningRate = learningRate;
        this.trainingHistory = new List<double>();
    }
    
    // Función de activación escalón (step function)
    private int StepActivation(double x)
    {
        return x >= 0 ? 1 : 0;
    }
    
    // La derivada del escalón es problemática, usamos una aproximación
    private double StepDerivative(double x)
    {
        // Para el escalón, usamos una aproximación práctica
        // En la práctica, se usa 1 cuando hay error para permitir actualización
        return 1.0;
    }
    
    // Entrenar el perceptrón
    public void Fit(double[,] X, int[] y, int epochs = 30)
    {
        int numSamples = X.GetLength(0);
        int numFeatures = X.GetLength(1);
        
        // Inicializar pesos aleatoriamente
        Random random = new Random();
        weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            weights[i] = (random.NextDouble() - 0.5) * 0.2;
        }
        bias = (random.NextDouble() - 0.5) * 0.2;
        
        // Preparar log de entrenamiento
        var logContent = new StringBuilder();
        logContent.AppendLine("=== ENTRENAMIENTO PERCEPTRON RIESGO ACADÉMICO ===");
        logContent.AppendLine("Función de activación: Escalón (Step)");
        logContent.AppendLine($"Tasa de aprendizaje: {learningRate}");
        logContent.AppendLine($"Épocas: {epochs}");
        logContent.AppendLine($"Fecha: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        logContent.AppendLine();
        
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            int updates = 0;
            
            for (int i = 0; i < numSamples; i++)
            {
                // Forward pass
                double linearOutput = DotProduct(GetRow(X, i), weights) + bias;
                int prediction = StepActivation(linearOutput);
                
                // Calcular error
                int error = y[i] - prediction;
                totalError += Math.Abs(error);
                
                // Actualizar solo si hay error (regla del perceptrón clásico)
                if (error != 0)
                {
                    updates++;
                    // Backward pass
                    for (int j = 0; j < weights.Length; j++)
                    {
                        weights[j] += learningRate * error * X[i, j];
                    }
                    bias += learningRate * error;
                }
            }
            
            double avgError = totalError / numSamples;
            trainingHistory.Add(avgError);
            
            if (epoch % 5 == 0)
            {
                string message = $"Época {epoch}, Error promedio: {avgError:F4}, Actualizaciones: {updates}";
                Console.WriteLine(message);
                logContent.AppendLine(message);
            }
            
            // Si no hay errores, el perceptrón ha convergido
            if (totalError == 0)
            {
                string convergenceMessage = $"Convergencia alcanzada en época {epoch}";
                Console.WriteLine(convergenceMessage);
                logContent.AppendLine(convergenceMessage);
                break;
            }
        }
        
        // Guardar pesos finales
        logContent.AppendLine();
        logContent.AppendLine("=== PESOS FINALES ===");
        for (int i = 0; i < weights.Length; i++)
        {
            logContent.AppendLine($"Peso {i}: {weights[i]:F4}");
        }
        logContent.AppendLine($"Bias: {bias:F4}");
        
        // Guardar log en archivo
        SaveToFile("resultados_riesgo_entrenamiento.txt", logContent.ToString());
    }
    
    // Método para guardar archivos
    public void SaveToFile(string filename, string content)
    {
        try
        {
            File.WriteAllText(filename, content, Encoding.UTF8);
            Console.WriteLine($"Resultados guardados en: {filename}");
        }
        catch (Exception e)
        {
            Console.WriteLine($"Error al guardar archivo: {e.Message}");
        }
    }
    
    // Realizar predicciones
    public PredictionResult Predict(double[,] X)
    {
        int numSamples = X.GetLength(0);
        int[] predictions = new int[numSamples];
        double[] scores = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++)
        {
            double linearOutput = DotProduct(GetRow(X, i), weights) + bias;
            int prediction = StepActivation(linearOutput);
            
            predictions[i] = prediction;
            scores[i] = linearOutput; // Score antes de la función escalón
        }
        
        return new PredictionResult(predictions, scores);
    }
    
    // Producto punto
    private double DotProduct(double[] a, double[] b)
    {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // Obtener fila de matriz
    private double[] GetRow(double[,] matrix, int row)
    {
        int cols = matrix.GetLength(1);
        double[] result = new double[cols];
        for (int i = 0; i < cols; i++)
        {
            result[i] = matrix[row, i];
        }
        return result;
    }
    
    public class PredictionResult
    {
        public int[] Predictions { get; }
        public double[] Scores { get; }
        
        public PredictionResult(int[] predictions, double[] scores)
        {
            Predictions = predictions;
            Scores = scores;
        }
    }
    
    public class DataSet
    {
        public double[,] X { get; }
        public int[] Y { get; }
        public double[,] XOriginal { get; }
        
        public DataSet(double[,] x, int[] y, double[,] xOriginal)
        {
            X = x;
            Y = y;
            XOriginal = xOriginal;
        }
    }
    
    // Generar datos para clasificación de riesgo académico
    public static DataSet GenerateAcademicRiskData()
    {
        // Características: [promedio_notas, asistencia_porcentaje, tareas_entregadas_porcentaje, participacion_clase, horas_estudio_semanal]
        double[,] XOriginal = new double[,]
        {
            {8.5, 95, 100, 8, 15},   // Sin riesgo
            {6.0, 60, 50, 3, 5},     // Con riesgo
            {9.0, 90, 95, 9, 20},    // Sin riesgo
            {5.5, 55, 40, 2, 3},     // Con riesgo
            {7.8, 85, 90, 7, 12},    // Sin riesgo
            {4.5, 45, 30, 1, 2},     // Con riesgo
            {8.2, 88, 85, 8, 14},    // Sin riesgo
            {5.8, 58, 45, 3, 4},     // Con riesgo
            {9.2, 92, 98, 9, 18},    // Sin riesgo
            {4.8, 48, 35, 2, 3},     // Con riesgo
            {7.5, 82, 88, 6, 11},    // Sin riesgo
            {5.2, 52, 42, 2, 4},     // Con riesgo
            {8.8, 89, 92, 8, 16},    // Sin riesgo
            {4.2, 42, 25, 1, 2},     // Con riesgo
            {7.2, 78, 80, 6, 10},    // Sin riesgo
            {5.0, 50, 38, 2, 3},     // Con riesgo
            {8.0, 86, 87, 7, 13},    // Sin riesgo
            {4.8, 47, 32, 1, 2},     // Con riesgo
            {9.5, 96, 100, 10, 22},  // Sin riesgo
            {3.8, 38, 20, 1, 1},     // Con riesgo
            {7.0, 75, 75, 5, 9},     // Sin riesgo
            {5.5, 55, 45, 3, 4},     // Con riesgo
            {8.7, 91, 94, 8, 17},    // Sin riesgo
            {4.0, 40, 28, 1, 2},     // Con riesgo
            {7.8, 80, 82, 6, 12},    // Sin riesgo
            {5.3, 53, 41, 2, 3},     // Con riesgo
            {8.4, 87, 89, 7, 15},    // Sin riesgo
            {4.6, 46, 33, 2, 2},     // Con riesgo
            {9.1, 93, 97, 9, 19},    // Sin riesgo
            {3.5, 35, 18, 1, 1}      // Con riesgo
        };
        
        int[] y = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
        
        // Normalizar características
        double[,] XNormalized = NormalizeData(XOriginal);
        
        return new DataSet(XNormalized, y, XOriginal);
    }
    
    // Normalizar datos
    private static double[,] NormalizeData(double[,] X)
    {
        int rows = X.GetLength(0);
        int cols = X.GetLength(1);
        
        // Calcular medias
        double[] means = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                means[j] += X[i, j];
            }
            means[j] /= rows;
        }
        
        // Calcular desviaciones estándar
        double[] stds = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                stds[j] += Math.Pow(X[i, j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / rows);
            if (stds[j] == 0) stds[j] = 1; // Evitar división por cero
        }
        
        // Normalizar
        double[,] XNormalized = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                XNormalized[i, j] = (X[i, j] - means[j]) / stds[j];
            }
        }
        
        return XNormalized;
    }
    
    public static void Main(string[] args)
    {
        Console.WriteLine("=== PERCEPTRÓN PARA CLASIFICACIÓN DE RIESGO ACADÉMICO ===");
        Console.WriteLine("Función de activación: Escalón (Step Function)");
        Console.WriteLine();
        
        // Generar datos
        DataSet data = GenerateAcademicRiskData();
        
        // Crear y entrenar el perceptrón
        PerceptronRiesgoAcademico perceptron = new PerceptronRiesgoAcademico(0.1);
        perceptron.Fit(data.X, data.Y, 40);
        
        // Realizar predicciones
        PredictionResult result = perceptron.Predict(data.X);
        
        // Calcular precisión
        int correct = 0;
        for (int i = 0; i < result.Predictions.Length; i++)
        {
            if (result.Predictions[i] == data.Y[i])
            {
                correct++;
            }
        }
        double accuracy = (double)correct / result.Predictions.Length * 100;
        
        Console.WriteLine($"\nPrecisión en datos de entrenamiento: {accuracy:F2}%");
        
        // Preparar resultados para archivo
        var resultsContent = new StringBuilder();
        resultsContent.AppendLine("=== RESULTADOS CLASIFICACIÓN RIESGO ACADÉMICO ===");
        resultsContent.AppendLine("Función de activación: Escalón (Step Function)");
        resultsContent.AppendLine($"Precisión: {accuracy:F2}%");
        resultsContent.AppendLine($"Fecha: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        resultsContent.AppendLine();
        
        // Mostrar algunos ejemplos
        Console.WriteLine("\n=== EJEMPLOS DE CLASIFICACIÓN ===");
        Console.WriteLine("Prom | Asist | Tareas | Partic | Horas | Real        | Predicción  | Score  | ¿Correcto?");
        Console.WriteLine("-----+-------+--------+--------+-------+-------------+-------------+--------+-----------");
        
        resultsContent.AppendLine("=== EJEMPLOS DE CLASIFICACIÓN ===");
        resultsContent.AppendLine("Prom | Asist | Tareas | Partic | Horas | Real        | Predicción  | Score  | ¿Correcto?");
        resultsContent.AppendLine("-----+-------+--------+--------+-------+-------------+-------------+--------+-----------");
        
        for (int i = 0; i < Math.Min(20, result.Predictions.Length); i++)
        {
            string realLabel = data.Y[i] == 1 ? "Con Riesgo " : "Sin Riesgo ";
            string predLabel = result.Predictions[i] == 1 ? "Con Riesgo " : "Sin Riesgo ";
            string correct_symbol = result.Predictions[i] == data.Y[i] ? "✓" : "✗";
            
            string line = $"{data.XOriginal[i, 0]:F1}  |  {data.XOriginal[i, 1]:F0}   |   {data.XOriginal[i, 2]:F0}   |   {data.XOriginal[i, 3]:F0}    |  {data.XOriginal[i, 4]:F0}   | {realLabel}  | {predLabel}  | {result.Scores[i]:F3}  |     {correct_symbol}";
            Console.WriteLine(line);
            resultsContent.AppendLine(line);
        }
        
        // Probar con nuevos estudiantes
        Console.WriteLine("\n=== PREDICCIONES PARA NUEVOS ESTUDIANTES ===");
        double[,] nuevosEstudiantes = new double[,]
        {
            {3.0, 30, 15, 0, 1},     // Muy alto riesgo
            {9.8, 98, 100, 10, 25},  // Sin riesgo
            {6.5, 70, 65, 4, 7},     // Riesgo moderado
            {4.5, 45, 35, 2, 3}      // Con riesgo
        };
        
        // Normalizar con los mismos parámetros (aproximación)
        double[,] nuevosNormalizados = NormalizeData(nuevosEstudiantes);
        PredictionResult nuevasPredicciones = perceptron.Predict(nuevosNormalizados);
        
        Console.WriteLine("Prom | Asist | Tareas | Partic | Horas | Predicción  | Score  | Recomendación");
        Console.WriteLine("-----+-------+--------+--------+-------+-------------+--------+------------------");
        
        resultsContent.AppendLine();
        resultsContent.AppendLine("=== PREDICCIONES PARA NUEVOS ESTUDIANTES ===");
        resultsContent.AppendLine("Prom | Asist | Tareas | Partic | Horas | Predicción  | Score  | Recomendación");
        resultsContent.AppendLine("-----+-------+--------+--------+-------+-------------+--------+------------------");
        
        for (int i = 0; i < nuevosEstudiantes.GetLength(0); i++)
        {
            string predLabel = nuevasPredicciones.Predictions[i] == 1 ? "Con Riesgo " : "Sin Riesgo ";
            string recomendacion = nuevasPredicciones.Predictions[i] == 1 ? 
                "Requiere intervención" : "Continuar monitoreo";
            
            string line = $"{nuevosEstudiantes[i, 0]:F1}  |  {nuevosEstudiantes[i, 1]:F0}   |   {nuevosEstudiantes[i, 2]:F0}   |   {nuevosEstudiantes[i, 3]:F0}    |  {nuevosEstudiantes[i, 4]:F0}   | {predLabel}  | {nuevasPredicciones.Scores[i]:F3}  | {recomendacion}";
            Console.WriteLine(line);
            resultsContent.AppendLine(line);
        }
        
        // Calcular estadísticas del modelo
        Console.WriteLine("\n=== ESTADÍSTICAS DEL MODELO ===");
        int truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
        
        for (int i = 0; i < result.Predictions.Length; i++)
        {
            if (result.Predictions[i] == 1 && data.Y[i] == 1) truePositives++;
            else if (result.Predictions[i] == 1 && data.Y[i] == 0) falsePositives++;
            else if (result.Predictions[i] == 0 && data.Y[i] == 0) trueNegatives++;
            else if (result.Predictions[i] == 0 && data.Y[i] == 1) falseNegatives++;
        }
        
        double precision = truePositives + falsePositives > 0 ? 
            (double)truePositives / (truePositives + falsePositives) : 0;
        double recall = truePositives + falseNegatives > 0 ? 
            (double)truePositives / (truePositives + falseNegatives) : 0;
        double f1Score = precision + recall > 0 ? 
            2 * (precision * recall) / (precision + recall) : 0;
        
        Console.WriteLine($"Precisión: {precision:F3}");
        Console.WriteLine($"Recall: {recall:F3}");
        Console.WriteLine($"F1-Score: {f1Score:F3}");
        Console.WriteLine($"Verdaderos Positivos: {truePositives}");
        Console.WriteLine($"Falsos Positivos: {falsePositives}");
        Console.WriteLine($"Verdaderos Negativos: {trueNegatives}");
        Console.WriteLine($"Falsos Negativos: {falseNegatives}");
        
        resultsContent.AppendLine();
        resultsContent.AppendLine("=== ESTADÍSTICAS DEL MODELO ===");
        resultsContent.AppendLine($"Precisión: {precision:F3}");
        resultsContent.AppendLine($"Recall: {recall:F3}");
        resultsContent.AppendLine($"F1-Score: {f1Score:F3}");
        resultsContent.AppendLine($"Verdaderos Positivos: {truePositives}");
        resultsContent.AppendLine($"Falsos Positivos: {falsePositives}");
        resultsContent.AppendLine($"Verdaderos Negativos: {trueNegatives}");
        resultsContent.AppendLine($"Falsos Negativos: {falseNegatives}");
        
        Console.WriteLine("\n=== INFORMACIÓN SOBRE LA FUNCIÓN ESCALÓN ===");
        Console.WriteLine("La función escalón (step function) es la función de activación más simple:");
        Console.WriteLine("• f(x) = 1 si x ≥ 0, sino 0");
        Console.WriteLine("• Produce salidas binarias claras (0 o 1)");
        Console.WriteLine("• Es la función original del perceptrón de Rosenblatt");
        Console.WriteLine("• Ideal para problemas de clasificación binaria linealmente separables");
        Console.WriteLine("• Convergencia garantizada para datos linealmente separables");
        
        resultsContent.AppendLine();
        resultsContent.AppendLine("=== INFORMACIÓN SOBRE LA FUNCIÓN ESCALÓN ===");
        resultsContent.AppendLine("La función escalón (step function) es la función de activación más simple:");
        resultsContent.AppendLine("• f(x) = 1 si x ≥ 0, sino 0");
        resultsContent.AppendLine("• Produce salidas binarias claras (0 o 1)");
        resultsContent.AppendLine("• Es la función original del perceptrón de Rosenblatt");
        resultsContent.AppendLine("• Ideal para problemas de clasificación binaria linealmente separables");
        resultsContent.AppendLine("• Convergencia garantizada para datos linealmente separables");
        
        Console.WriteLine("\nEl perceptrón puede ayudar a identificar estudiantes en riesgo académico");
        Console.WriteLine("para intervención temprana y apoyo personalizado.");
        
        resultsContent.AppendLine();
        resultsContent.AppendLine("El perceptrón puede ayudar a identificar estudiantes en riesgo académico");
        resultsContent.AppendLine("para intervención temprana y apoyo personalizado.");
        
        // Guardar resultados completos
        perceptron.SaveToFile("resultados_riesgo_completos.txt", resultsContent.ToString());
    }
}
