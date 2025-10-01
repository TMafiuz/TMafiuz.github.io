import java.util.*;
import java.io.*;
import javax.swing.*;
import java.awt.*;

public class PerceptronAND {
    private double[] weights;
    private double bias;
    private double learningRate;
    private java.util.List<Double> trainingHistory;
    
    public PerceptronAND(double learningRate) {
        this.learningRate = learningRate;
        this.trainingHistory = new ArrayList<>();
    }
    
    // Función de activación lineal
    private double linearActivation(double x) {
        return x; // Función identidad
    }
    
    // Derivada de la función lineal (siempre es 1)
    private double linearDerivative(double x) {
        return 1.0;
    }
    
    // Entrenar el perceptrón
    public void fit(double[][] X, int[] y, int epochs) {
        int numFeatures = X[0].length;
        
        // Inicializar pesos aleatoriamente
        Random random = new Random();
        weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 0.2;
        }
        bias = (random.nextDouble() - 0.5) * 0.2;
        
        // Para el archivo de resultados
        StringBuilder log = new StringBuilder();
        log.append("=== ENTRENAMIENTO PERCEPTRON AND LÓGICO ===\n");
        log.append("Función de activación: Lineal\n");
        log.append("Tasa de aprendizaje: ").append(learningRate).append("\n");
        log.append("Épocas: ").append(epochs).append("\n\n");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            
            for (int i = 0; i < X.length; i++) {
                // Forward pass
                double linearOutput = dotProduct(X[i], weights) + bias;
                double prediction = linearActivation(linearOutput);
                
                // Convertir a clasificación binaria
                int binaryPrediction = prediction > 0.5 ? 1 : 0;
                
                // Calcular error
                double error = y[i] - binaryPrediction;
                totalError += error * error;
                
                // Backward pass
                double gradient = error * linearDerivative(linearOutput);
                
                // Actualizar pesos
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * gradient * X[i][j];
                }
                bias += learningRate * gradient;
            }
            
            double avgError = totalError / X.length;
            trainingHistory.add(avgError);
            
            if (epoch % 5 == 0) {
                String message = String.format("Época %d, Error promedio: %.4f", epoch, avgError);
                System.out.println(message);
                log.append(message).append("\n");
            }
        }
        
        // Guardar pesos finales
        log.append("\n=== PESOS FINALES ===\n");
        for (int i = 0; i < weights.length; i++) {
            log.append("Peso ").append(i).append(": ").append(String.format("%.4f", weights[i])).append("\n");
        }
        log.append("Bias: ").append(String.format("%.4f", bias)).append("\n");
        
        // Guardar log en archivo
        saveToFile("resultados_AND.txt", log.toString());
    }
    
    // Realizar predicciones
    public PredictionResult predict(double[][] X) {
        int[] predictions = new int[X.length];
        double[] scores = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double linearOutput = dotProduct(X[i], weights) + bias;
            double score = linearActivation(linearOutput);
            int prediction = score > 0.5 ? 1 : 0;
            
            predictions[i] = prediction;
            scores[i] = score;
        }
        
        return new PredictionResult(predictions, scores);
    }
    
    // Producto punto
    private double dotProduct(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // Guardar resultados en archivo
    public void saveToFile(String filename, String content) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.print(content);
            System.out.println("Resultados guardados en: " + filename);
        } catch (IOException e) {
            System.err.println("Error al guardar archivo: " + e.getMessage());
        }
    }
    
    // Mostrar gráfico del historial de entrenamiento
    public void showTrainingGraph() {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Historial de Entrenamiento - Perceptrón AND");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new TrainingGraphPanel(this.trainingHistory));
            frame.setSize(800, 600);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
    
    // Panel para el gráfico
    private static class TrainingGraphPanel extends JPanel {
        private java.util.List<Double> history;
        
        public TrainingGraphPanel(java.util.List<Double> history) {
            this.history = history;
            setBackground(Color.WHITE);
        }
        
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            
            int width = getWidth();
            int height = getHeight();
            int margin = 50;
            
            // Dibujar ejes
            g2d.setColor(Color.BLACK);
            g2d.drawLine(margin, height - margin, width - margin, height - margin); // X axis
            g2d.drawLine(margin, margin, margin, height - margin); // Y axis
            
            // Etiquetas
            g2d.drawString("Épocas", width / 2, height - 10);
            g2d.rotate(-Math.PI / 2);
            g2d.drawString("Error", -height / 2, 20);
            g2d.rotate(Math.PI / 2);
            
            if (history.isEmpty()) return;
            
            // Encontrar valores máximos
            double maxError = history.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
            int maxEpochs = history.size();
            
            // Dibujar línea del historial
            g2d.setColor(Color.BLUE);
            g2d.setStroke(new BasicStroke(2));
            
            for (int i = 0; i < history.size() - 1; i++) {
                int x1 = margin + (i * (width - 2 * margin)) / maxEpochs;
                int y1 = height - margin - (int)((history.get(i) / maxError) * (height - 2 * margin));
                int x2 = margin + ((i + 1) * (width - 2 * margin)) / maxEpochs;
                int y2 = height - margin - (int)((history.get(i + 1) / maxError) * (height - 2 * margin));
                
                g2d.drawLine(x1, y1, x2, y2);
            }
            
            // Dibujar puntos
            g2d.setColor(Color.RED);
            for (int i = 0; i < history.size(); i++) {
                int x = margin + (i * (width - 2 * margin)) / maxEpochs;
                int y = height - margin - (int)((history.get(i) / maxError) * (height - 2 * margin));
                g2d.fillOval(x - 3, y - 3, 6, 6);
            }
            
            // Dibujar escala
            g2d.setColor(Color.GRAY);
            for (int i = 0; i <= 5; i++) {
                int y = height - margin - (i * (height - 2 * margin)) / 5;
                g2d.drawString(String.format("%.2f", (maxError * i) / 5), 5, y);
                
                int x = margin + (i * (width - 2 * margin)) / 5;
                g2d.drawString(String.valueOf((maxEpochs * i) / 5), x, height - 10);
            }
        }
    }
    
    // Clase para el resultado de predicciones
    public static class PredictionResult {
        public int[] predictions;
        public double[] scores;
        
        public PredictionResult(int[] predictions, double[] scores) {
            this.predictions = predictions;
            this.scores = scores;
        }
    }
    
    // Generar datos para AND lógico
    public static DataSet generateANDData() {
        double[][] X = {
            {0, 0},
            {0, 1}, 
            {1, 0},
            {1, 1}
        };
        
        int[] y = {0, 0, 0, 1}; // AND lógico
        
        return new DataSet(X, y);
    }
    
    // Clase para el conjunto de datos
    public static class DataSet {
        public double[][] X;
        public int[] y;
        
        public DataSet(double[][] X, int[] y) {
            this.X = X;
            this.y = y;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== PERCEPTRÓN AND LÓGICO ===");
        System.out.println("Función de activación: Lineal");
        System.out.println();
        
        // Generar datos
        DataSet data = generateANDData();
        
        // Crear y entrenar el perceptrón
        PerceptronAND perceptron = new PerceptronAND(0.1);
        perceptron.fit(data.X, data.y, 50);
        
        // Realizar predicciones
        PredictionResult result = perceptron.predict(data.X);
        
        // Calcular precisión
        int correct = 0;
        for (int i = 0; i < result.predictions.length; i++) {
            if (result.predictions[i] == data.y[i]) {
                correct++;
            }
        }
        double accuracy = (double) correct / result.predictions.length * 100;
        
        System.out.printf("%nPrecisión: %.2f%%%n", accuracy);
        
        // Preparar resultados para archivo
        StringBuilder resultLog = new StringBuilder();
        resultLog.append("\n=== RESULTADOS DE PREDICCIÓN ===\n");
        resultLog.append(String.format("Precisión: %.2f%%\n\n", accuracy));
        
        // Mostrar tabla de verdad
        System.out.println("\n=== TABLA DE VERDAD AND ===");
        System.out.println("A | B | Esperado | Predicción | Puntuación | ¿Correcto?");
        System.out.println("--+---+----------+------------+------------+-----------");
        
        resultLog.append("=== TABLA DE VERDAD AND ===\n");
        resultLog.append("A | B | Esperado | Predicción | Puntuación | ¿Correcto?\n");
        resultLog.append("--+---+----------+------------+------------+-----------\n");
        
        for (int i = 0; i < data.X.length; i++) {
            String correct_symbol = result.predictions[i] == data.y[i] ? "✓" : "✗";
            String line = String.format("%.0f | %.0f |    %d     |     %d      |   %.3f    |     %s",
                data.X[i][0], data.X[i][1], data.y[i], 
                result.predictions[i], result.scores[i], correct_symbol);
            System.out.println(line);
            resultLog.append(line).append("\n");
        }
        
        // Probar con valores adicionales
        System.out.println("\n=== PRUEBAS ADICIONALES ===");
        double[][] testData = {
            {0.1, 0.1},  // Debería ser 0
            {0.9, 0.1},  // Debería ser 0
            {0.1, 0.9},  // Debería ser 0
            {0.9, 0.9}   // Debería ser 1
        };
        
        PredictionResult testResult = perceptron.predict(testData);
        
        System.out.println("A   | B   | Predicción | Puntuación");
        System.out.println("----+-----+------------+-----------");
        
        resultLog.append("\n=== PRUEBAS ADICIONALES ===\n");
        resultLog.append("A   | B   | Predicción | Puntuación\n");
        resultLog.append("----+-----+------------+-----------\n");
        
        for (int i = 0; i < testData.length; i++) {
            String line = String.format("%.1f | %.1f |     %d      |   %.3f",
                testData[i][0], testData[i][1], 
                testResult.predictions[i], testResult.scores[i]);
            System.out.println(line);
            resultLog.append(line).append("\n");
        }
        
        resultLog.append("\nEl perceptrón con función lineal ha aprendido la función AND lógica!\n");
        resultLog.append("La función lineal f(x) = x permite que el perceptrón aprenda\n");
        resultLog.append("mediante ajuste directo de pesos sin transformación no lineal.\n");
        
        System.out.println("\nEl perceptrón con función lineal ha aprendido la función AND lógica!");
        System.out.println("La función lineal f(x) = x permite que el perceptrón aprenda");
        System.out.println("mediante ajuste directo de pesos sin transformación no lineal.");
        
        // Guardar resultados adicionales
        perceptron.saveToFile("resultados_AND_completos.txt", resultLog.toString());
        
        // Mostrar gráfico
        perceptron.showTrainingGraph();
    }
}
