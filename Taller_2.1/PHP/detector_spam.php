<?php

class PerceptronSpam {
    private $weights;
    private $bias;
    private $learning_rate;
    private $training_history;
    
    public function __construct($learning_rate = 0.01) {
        $this->learning_rate = $learning_rate;
        $this->weights = null;
        $this->bias = 0;
        $this->training_history = [];
    }
    
    // Función de activación ReLU
    private function relu_activation($x) {
        return max(0, $x);
    }
    
    // Derivada de ReLU
    private function relu_derivative($x) {
        return $x > 0 ? 1 : 0;
    }
    
    // Entrenar el perceptrón
    public function fit($X, $y, $epochs = 50) {
        $num_features = count($X[0]);
        
        // Inicializar pesos aleatoriamente
        $this->weights = [];
        for ($i = 0; $i < $num_features; $i++) {
            $this->weights[] = (mt_rand() / mt_getrandmax() - 0.5) * 0.2;
        }
        $this->bias = (mt_rand() / mt_getrandmax() - 0.5) * 0.2;
        
        // Preparar log de entrenamiento
        $log_content = [];
        $log_content[] = "=== ENTRENAMIENTO PERCEPTRON DETECTOR SPAM ===";
        $log_content[] = "Función de activación: ReLU";
        $log_content[] = "Tasa de aprendizaje: " . $this->learning_rate;
        $log_content[] = "Épocas: " . $epochs;
        $log_content[] = "Fecha: " . date('Y-m-d H:i:s');
        $log_content[] = "";
        
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $total_error = 0;
            
            for ($i = 0; $i < count($X); $i++) {
                // Forward pass
                $linear_output = $this->dot_product($X[$i], $this->weights) + $this->bias;
                $prediction = $this->relu_activation($linear_output);
                
                // Convertir a clasificación binaria para ReLU
                $binary_prediction = $prediction > 0.5 ? 1 : 0;
                
                // Calcular error
                $error = $y[$i] - $binary_prediction;
                $total_error += $error * $error;
                
                // Backward pass
                if ($linear_output > 0) { // Solo actualizar si ReLU está activa
                    $gradient = $error * $this->relu_derivative($linear_output);
                    
                    // Actualizar pesos
                    for ($j = 0; $j < count($this->weights); $j++) {
                        $this->weights[$j] += $this->learning_rate * $gradient * $X[$i][$j];
                    }
                    $this->bias += $this->learning_rate * $gradient;
                }
            }
            
            $avg_error = $total_error / count($X);
            $this->training_history[] = $avg_error;
            
            if ($epoch % 10 == 0) {
                $message = "Época $epoch, Error promedio: " . number_format($avg_error, 4);
                echo $message . "\n";
                $log_content[] = $message;
            }
        }
        
        // Guardar pesos finales
        $log_content[] = "";
        $log_content[] = "=== PESOS FINALES ===";
        for ($i = 0; $i < count($this->weights); $i++) {
            $log_content[] = "Peso $i: " . number_format($this->weights[$i], 4);
        }
        $log_content[] = "Bias: " . number_format($this->bias, 4);
        
        // Guardar log en archivo
        $this->save_to_file("resultados_spam_entrenamiento.txt", implode("\n", $log_content));
        
        // Devolver el log para uso en la interfaz web
        return implode("\n", $log_content);
    }
    
    // Método para guardar archivos
    public function save_to_file($filename, $content) {
        try {
            file_put_contents($filename, $content);
            echo "Resultados guardados en: $filename\n";
        } catch (Exception $e) {
            echo "Error al guardar archivo: " . $e->getMessage() . "\n";
        }
    }
    
    // Realizar predicciones
    public function predict($X) {
        $predictions = [];
        $scores = [];
        
        foreach ($X as $x) {
            $linear_output = $this->dot_product($x, $this->weights) + $this->bias;
            $score = $this->relu_activation($linear_output);
            $prediction = $score > 0.5 ? 1 : 0;
            
            $predictions[] = $prediction;
            $scores[] = $score;
        }
        
        return ['predictions' => $predictions, 'scores' => $scores];
    }
    
    // Producto punto de dos vectores
    private function dot_product($a, $b) {
        $result = 0;
        for ($i = 0; $i < count($a); $i++) {
            $result += $a[$i] * $b[$i];
        }
        return $result;
    }
    
    // Obtener historia del entrenamiento
    public function get_training_history() {
        return $this->training_history;
    }
    
    // Obtener pesos del modelo
    public function get_weights() {
        return $this->weights;
    }
    
    // Obtener bias del modelo
    public function get_bias() {
        return $this->bias;
    }
}

// Generar datos de entrenamiento para detección de spam
function generar_datos_spam() {
    // Características: [num_palabras_spam, num_mayusculas, num_signos_exclamacion, longitud_email, tiene_enlaces]
    $X = [
        [0, 2, 0, 50, 0],    // No spam
        [8, 15, 5, 200, 1],  // Spam
        [1, 3, 0, 75, 0],    // No spam
        [12, 20, 8, 300, 1], // Spam
        [0, 1, 0, 40, 0],    // No spam
        [15, 25, 10, 250, 1], // Spam
        [2, 4, 1, 80, 0],    // No spam
        [10, 18, 6, 280, 1], // Spam
        [1, 2, 0, 60, 0],    // No spam
        [14, 22, 9, 320, 1], // Spam
        [0, 3, 0, 45, 0],    // No spam
        [9, 16, 7, 240, 1],  // Spam
        [2, 5, 1, 85, 0],    // No spam
        [13, 24, 11, 290, 1], // Spam
        [1, 1, 0, 55, 0],    // No spam
        [11, 19, 8, 270, 1], // Spam
        [0, 2, 0, 65, 0],    // No spam
        [16, 26, 12, 310, 1], // Spam
        [2, 4, 1, 70, 0],    // No spam
        [8, 17, 6, 260, 1],  // Spam
        [1, 3, 0, 48, 0],    // No spam
        [17, 28, 13, 330, 1], // Spam
        [0, 1, 0, 42, 0],    // No spam
        [7, 14, 5, 220, 1],  // Spam
        [2, 6, 1, 78, 0],    // No spam
        [18, 30, 15, 350, 1], // Spam
        [1, 2, 0, 52, 0],    // No spam
        [6, 13, 4, 210, 1],  // Spam
        [0, 4, 0, 68, 0],    // No spam
        [19, 32, 16, 380, 1]  // Spam
    ];
    
    $y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    
    // Normalizar características
    $X_normalized = normalizar_datos($X);
    
    return ['X' => $X_normalized, 'y' => $y, 'X_original' => $X];
}

// Función para normalizar datos
function normalizar_datos($X) {
    $num_samples = count($X);
    $num_features = count($X[0]);
    
    // Calcular medias
    $means = array_fill(0, $num_features, 0);
    for ($i = 0; $i < $num_samples; $i++) {
        for ($j = 0; $j < $num_features; $j++) {
            $means[$j] += $X[$i][$j];
        }
    }
    for ($j = 0; $j < $num_features; $j++) {
        $means[$j] /= $num_samples;
    }
    
    // Calcular desviaciones estándar
    $stds = array_fill(0, $num_features, 0);
    for ($i = 0; $i < $num_samples; $i++) {
        for ($j = 0; $j < $num_features; $j++) {
            $stds[$j] += pow($X[$i][$j] - $means[$j], 2);
        }
    }
    for ($j = 0; $j < $num_features; $j++) {
        $stds[$j] = sqrt($stds[$j] / $num_samples);
        if ($stds[$j] == 0) $stds[$j] = 1; // Evitar división por cero
    }
    
    // Normalizar
    $X_normalized = [];
    for ($i = 0; $i < $num_samples; $i++) {
        $X_normalized[$i] = [];
        for ($j = 0; $j < $num_features; $j++) {
            $X_normalized[$i][$j] = ($X[$i][$j] - $means[$j]) / $stds[$j];
        }
    }
    
    return $X_normalized;
}

// Manejar la solicitud del formulario
$result = null;
$training_result = null;

// Manejar descarga de resultados
if (isset($_GET['download'])) {
    session_start();
    
    if ($_GET['download'] === 'results' && isset($_SESSION['training_results'])) {
        $results_content = $_SESSION['training_results'];
        
        header('Content-Type: text/plain; charset=utf-8');
        header('Content-Disposition: attachment; filename="resultados_spam_detallados_' . date('Y-m-d_H-i-s') . '.txt"');
        header('Content-Length: ' . strlen($results_content));
        
        echo $results_content;
        exit;
    }
    
    if ($_GET['download'] === 'training' && isset($_SESSION['training_log'])) {
        $training_log = $_SESSION['training_log'];
        
        header('Content-Type: text/plain; charset=utf-8');
        header('Content-Disposition: attachment; filename="entrenamiento_spam_log_' . date('Y-m-d_H-i-s') . '.txt"');
        header('Content-Length: ' . strlen($training_log));
        
        echo $training_log;
        exit;
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['train'])) {
        // Entrenar el modelo
        $data = generar_datos_spam();
        $X = $data['X'];
        $y = $data['y'];
        $X_original = $data['X_original'];
        
        $perceptron = new PerceptronSpam(0.02);
        $training_log = $perceptron->fit($X, $y, 60);
        
        $predictions_data = $perceptron->predict($X);
        $predictions = $predictions_data['predictions'];
        $scores = $predictions_data['scores'];
        
        $accuracy = 0;
        for ($i = 0; $i < count($predictions); $i++) {
            if ($predictions[$i] == $y[$i]) $accuracy++;
        }
        $accuracy = ($accuracy / count($predictions)) * 100;
        
        // Preparar resultados para archivo
        $results_content = [];
        $results_content[] = "=== RESULTADOS DETECTOR DE SPAM ===";
        $results_content[] = "Función de activación: ReLU";
        $results_content[] = "Precisión: " . number_format($accuracy, 2) . "%";
        $results_content[] = "Fecha: " . date('Y-m-d H:i:s');
        $results_content[] = "";
        $results_content[] = "=== EJEMPLOS DE PREDICCIÓN ===";
        $results_content[] = "Palabras | Mayúsc | Exclam | Longit | Enlaces | Real     | Predicción | Score | ¿Correcto?";
        $results_content[] = "---------|--------|--------|--------|---------|----------|------------|-------|----------";
        
        for ($i = 0; $i < count($predictions); $i++) {
            $real_label = $y[$i] ? 'Spam' : 'Legítimo';
            $pred_label = $predictions[$i] ? 'Spam' : 'Legítimo';
            $correct = $predictions[$i] == $y[$i] ? '✓' : '✗';
            $results_content[] = sprintf("%8d | %6d | %6d | %6d | %7s | %8s | %10s | %5.3f | %9s",
                $X_original[$i][0], $X_original[$i][1], $X_original[$i][2], 
                $X_original[$i][3], $X_original[$i][4] ? 'Sí' : 'No',
                $real_label, $pred_label, $scores[$i], $correct);
        }
        
        $results_content[] = "";
        $results_content[] = "=== INFORMACIÓN SOBRE RELU ===";
        $results_content[] = "ReLU (Rectified Linear Unit) es una función de activación que:";
        $results_content[] = "- Función: f(x) = max(0, x)";
        $results_content[] = "- Computacionalmente eficiente";
        $results_content[] = "- Evita el problema de gradientes que desaparecen";
        $results_content[] = "- Devuelve 0 para valores negativos y x para valores positivos";
        $results_content[] = "- Para clasificación binaria se usa umbral en 0.5";
        
        // Guardar resultados completos
        $perceptron->save_to_file("resultados_spam_completos.txt", implode("\n", $results_content));
        
        $training_result = [
            'perceptron' => $perceptron,
            'X_original' => $X_original,
            'y' => $y,
            'predictions' => $predictions,
            'scores' => $scores,
            'accuracy' => $accuracy
        ];
        
        // Guardar el modelo en sesión para uso posterior
        session_start();
        $_SESSION['trained_model'] = serialize($perceptron);
        $_SESSION['training_stats'] = serialize([
            'means' => $data['means'] ?? [],
            'stds' => $data['stds'] ?? []
        ]);
        // Guardar resultados para descarga
        $_SESSION['training_results'] = implode("\n", $results_content);
        $_SESSION['training_log'] = $training_log;
    }
    
    if (isset($_POST['predict'])) {
        session_start();
        if (isset($_SESSION['trained_model'])) {
            $perceptron = unserialize($_SESSION['trained_model']);
            
            // Obtener datos del formulario
            $num_palabras_spam = intval($_POST['num_palabras_spam']);
            $num_mayusculas = intval($_POST['num_mayusculas']);
            $num_signos_exclamacion = intval($_POST['num_signos_exclamacion']);
            $longitud_email = intval($_POST['longitud_email']);
            $tiene_enlaces = intval($_POST['tiene_enlaces']);
            
            // Normalizar los datos de entrada (usando estadísticas aproximadas)
            $input_data = [
                ($num_palabras_spam - 8.5) / 6.5,
                ($num_mayusculas - 13.5) / 10.0,
                ($num_signos_exclamacion - 6.0) / 5.0,
                ($longitud_email - 180) / 100,
                ($tiene_enlaces - 0.5) / 0.5
            ];
            
            $prediction_data = $perceptron->predict([$input_data]);
            $prediction = $prediction_data['predictions'][0];
            $score = $prediction_data['scores'][0];
            
            $result = [
                'prediction' => $prediction,
                'score' => $score,
                'input' => [
                    'num_palabras_spam' => $num_palabras_spam,
                    'num_mayusculas' => $num_mayusculas,
                    'num_signos_exclamacion' => $num_signos_exclamacion,
                    'longitud_email' => $longitud_email,
                    'tiene_enlaces' => $tiene_enlaces
                ]
            ];
        }
    }
}
?>

<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perceptrón Detector de Spam - ReLU</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
            margin-right: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn.download {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        
        .btn.download:hover {
            background: linear-gradient(135deg, #218838 0%, #1ea986 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .spam {
            background-color: #ffebee;
            color: #c62828;
            border-left: 4px solid #c62828;
        }
        
        .no-spam {
            background-color: #e8f5e8;
            color: #2e7d32;
            border-left: 4px solid #2e7d32;
        }
        
        .training-results {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        
        .examples-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .examples-table th, .examples-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .examples-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .examples-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .accuracy {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            text-align: center;
            margin: 20px 0;
        }
        
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Perceptrón Detector de Spam</h1>
        <p>Función de Activación: ReLU (Rectified Linear Unit)</p>
        <p>Clasificación inteligente de correos electrónicos</p>
    </div>

    <div class="container">
        <div class="card">
            <h3>🎯 Entrenar Modelo</h3>
            <div class="info-box">
                <strong>Función ReLU:</strong> f(x) = max(0, x)<br>
                <strong>Características analizadas:</strong> Palabras spam, mayúsculas, signos de exclamación, longitud y enlaces
            </div>
            <form method="post">
                <button type="submit" name="train" class="btn">🚀 Entrenar Perceptrón</button>
            </form>
        </div>

        <div class="card">
            <h3>📧 Analizar Email</h3>
            <form method="post">
                <div class="form-group">
                    <label for="num_palabras_spam">Número de palabras spam detectadas:</label>
                    <input type="number" id="num_palabras_spam" name="num_palabras_spam" min="0" max="50" value="5" required>
                </div>

                <div class="form-group">
                    <label for="num_mayusculas">Número de letras mayúsculas:</label>
                    <input type="number" id="num_mayusculas" name="num_mayusculas" min="0" max="100" value="10" required>
                </div>

                <div class="form-group">
                    <label for="num_signos_exclamacion">Número de signos de exclamación:</label>
                    <input type="number" id="num_signos_exclamacion" name="num_signos_exclamacion" min="0" max="20" value="3" required>
                </div>

                <div class="form-group">
                    <label for="longitud_email">Longitud del email (caracteres):</label>
                    <input type="number" id="longitud_email" name="longitud_email" min="10" max="1000" value="150" required>
                </div>

                <div class="form-group">
                    <label for="tiene_enlaces">¿Contiene enlaces sospechosos?</label>
                    <select id="tiene_enlaces" name="tiene_enlaces" required>
                        <option value="0">No</option>
                        <option value="1">Sí</option>
                    </select>
                </div>

                <button type="submit" name="predict" class="btn">🔍 Analizar Email</button>
            </form>
        </div>
    </div>

    <?php if ($result): ?>
    <div class="card">
        <h3>📊 Resultado del Análisis</h3>
        <div class="result <?php echo $result['prediction'] == 1 ? 'spam' : 'no-spam'; ?>">
            <?php if ($result['prediction'] == 1): ?>
                🚨 <strong>SPAM DETECTADO</strong> - Este email parece ser spam
            <?php else: ?>
                ✅ <strong>EMAIL LEGÍTIMO</strong> - Este email parece ser auténtico
            <?php endif; ?>
        </div>
        
        <div style="margin-top: 15px;">
            <strong>Puntuación ReLU:</strong> <?php echo number_format($result['score'], 3); ?><br>
            <strong>Características analizadas:</strong>
            <ul>
                <li>Palabras spam: <?php echo $result['input']['num_palabras_spam']; ?></li>
                <li>Mayúsculas: <?php echo $result['input']['num_mayusculas']; ?></li>
                <li>Signos de exclamación: <?php echo $result['input']['num_signos_exclamacion']; ?></li>
                <li>Longitud: <?php echo $result['input']['longitud_email']; ?> caracteres</li>
                <li>Enlaces sospechosos: <?php echo $result['input']['tiene_enlaces'] ? 'Sí' : 'No'; ?></li>
            </ul>
        </div>
    </div>
    <?php endif; ?>

    <?php if ($training_result): ?>
    <div class="training-results">
        <h3>📈 Resultados del Entrenamiento</h3>
        <div class="accuracy">
            Precisión del Modelo: <?php echo number_format($training_result['accuracy'], 2); ?>%
        </div>
        
        <h4>Ejemplos de Clasificación (Primeros 15):</h4>
        <table class="examples-table">
            <thead>
                <tr>
                    <th>Palabras Spam</th>
                    <th>Mayúsculas</th>
                    <th>Exclamaciones</th>
                    <th>Longitud</th>
                    <th>Enlaces</th>
                    <th>Real</th>
                    <th>Predicción</th>
                    <th>Puntuación</th>
                    <th>Estado</th>
                </tr>
            </thead>
            <tbody>
                <?php for ($i = 0; $i < min(15, count($training_result['predictions'])); $i++): ?>
                <tr>
                    <td><?php echo $training_result['X_original'][$i][0]; ?></td>
                    <td><?php echo $training_result['X_original'][$i][1]; ?></td>
                    <td><?php echo $training_result['X_original'][$i][2]; ?></td>
                    <td><?php echo $training_result['X_original'][$i][3]; ?></td>
                    <td><?php echo $training_result['X_original'][$i][4] ? 'Sí' : 'No'; ?></td>
                    <td><?php echo $training_result['y'][$i] ? 'Spam' : 'Legítimo'; ?></td>
                    <td><?php echo $training_result['predictions'][$i] ? 'Spam' : 'Legítimo'; ?></td>
                    <td><?php echo number_format($training_result['scores'][$i], 3); ?></td>
                    <td><?php echo $training_result['predictions'][$i] == $training_result['y'][$i] ? '✅' : '❌'; ?></td>
                </tr>
                <?php endfor; ?>
            </tbody>
        </table>
        
        <div class="info-box" style="margin-top: 20px;">
            <strong>Información del Modelo:</strong><br>
            • Función de activación: ReLU (f(x) = max(0, x))<br>
            • Épocas de entrenamiento: 60<br>
            • Tasa de aprendizaje: 0.02<br>
            • Características: 5 (palabras spam, mayúsculas, exclamaciones, longitud, enlaces)<br>
            • Total de muestras de entrenamiento: <?php echo count($training_result['y']); ?>
        </div>
        
        <div style="margin-top: 20px; text-align: center;">
            <a href="?download=results" class="btn download" style="text-decoration: none; display: inline-block;">
                📥 Descargar Resultados Completos
            </a>
            <a href="?download=training" class="btn download" style="text-decoration: none; display: inline-block; margin-left: 10px;">
                📊 Descargar Log de Entrenamiento
            </a>
            <p style="margin-top: 10px; font-size: 12px; color: #666;">
                Archivos incluyen estadísticas detalladas, pesos del modelo e información sobre ReLU
            </p>
        </div>
    </div>
    <?php endif; ?>

    <div class="card" style="margin-top: 30px;">
        <h3>ℹ️ Información sobre ReLU</h3>
        <p><strong>ReLU (Rectified Linear Unit)</strong> es una función de activación muy popular en redes neuronales:</p>
        <ul>
            <li><strong>Función:</strong> f(x) = max(0, x)</li>
            <li><strong>Ventajas:</strong> Computacionalmente eficiente, evita el problema de gradientes que desaparecen</li>
            <li><strong>Comportamiento:</strong> Devuelve 0 para valores negativos y el mismo valor para valores positivos</li>
            <li><strong>Uso en clasificación:</strong> La salida se convierte a binario usando un umbral (0.5)</li>
        </ul>
    </div>
</body>
</html>
