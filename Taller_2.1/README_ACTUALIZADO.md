# Implementación de Perceptrones con Diferentes Funciones de Activación

Este proyecto implementa 6 perceptrones diferentes en distintos lenguajes de programación, cada uno con una función de activación específica y aplicado a casos de uso particulares.

## 🎯 Estructura del Proyecto

```
Taller 1/
├── Python/
│   ├── prediccion_clima.py          # Tangente Hiperbólica
│   ├── deteccion_fraude.py          # Sigmoidal
│   ├── grafico_entrenamiento_clima.png
│   └── grafico_entrenamiento_fraude.png
├── Java/
│   ├── PerceptronAND.java           # Función Lineal
│   └── PerceptronOR.java            # Softmax
├── CSharp/
│   └── PerceptronRiesgoAcademico.cs # Función Escalón
├── PHP/
│   ├── detector_spam.php            # ReLU + Vista HTML
│   └── vista_spam.html
├── Resultados/
│   ├── *.txt                        # Archivos de resultados
│   └── *.png                        # Gráficos generados
├── ejecutar_todos.bat               # Script para Windows
├── ejecutar_todos.sh                # Script para Linux/Mac
└── README.md
```

## 🚀 Nuevas Funcionalidades

### 📊 Gráficos Interactivos
- **Java**: Gráficos Swing que muestran el historial de entrenamiento
- **Python**: Gráficos matplotlib con subplots de función de activación
- **PHP**: Interfaz web HTML con resultados visuales

### 💾 Guardado de Resultados
Todos los perceptrones ahora guardan automáticamente:
- **Archivos de entrenamiento**: `resultados_*_entrenamiento.txt`
- **Resultados completos**: `resultados_*_completos.txt`
- **Gráficos**: Archivos PNG (Python)

### 📈 Información Detallada
Cada resultado incluye:
- Parámetros de entrenamiento
- Historial de épocas
- Pesos finales
- Métricas de rendimiento
- Información sobre la función de activación
- Ejemplos de predicción
- Recomendaciones específicas del dominio

## 🎮 Casos de Uso Implementados

| Lenguaje | Caso de Uso | Función de Activación | Características |
|----------|-------------|----------------------|-----------------|
| **Python** | Predicción del Clima | Tangente Hiperbólica | Temperatura, Humedad, Presión, Viento |
| **Python** | Detección de Fraude | Sigmoidal | Monto, Hora, Ubicación, Frecuencia, Edad cuenta |
| **Java** | AND Lógico | Lineal | Dos entradas binarias |
| **Java** | OR Lógico | Softmax | Dos entradas binarias |
| **C#** | Riesgo Académico | Escalón | Promedio, Asistencia, Tareas, Participación, Horas |
| **PHP** | Detector de Spam | ReLU | Palabras spam, Mayúsculas, Exclamaciones, Longitud, Enlaces |

## 🔧 Cómo Ejecutar

### Ejecución Individual

#### Python
```bash
cd Python
python prediccion_clima.py
python deteccion_fraude.py
```

#### Java
```bash
cd Java
javac PerceptronAND.java
java PerceptronAND
javac PerceptronOR.java
java PerceptronOR
```

#### C#
```bash
cd CSharp
csc PerceptronRiesgoAcademico.cs
PerceptronRiesgoAcademico.exe
```

#### PHP
```bash
cd PHP
php detector_spam.php
# O abrir en navegador para la interfaz web
```

### Ejecución Automática

#### Windows
```bash
ejecutar_todos.bat
```

#### Linux/Mac
```bash
chmod +x ejecutar_todos.sh
./ejecutar_todos.sh
```

## 📊 Funciones de Activación

### 1. **Tangente Hiperbólica** (Python - Clima)
- **Fórmula**: `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`
- **Rango**: (-1, 1)
- **Ventajas**: Simétrica, derivada fácil, centra salidas en 0

### 2. **Sigmoidal** (Python - Fraude)
- **Fórmula**: `σ(x) = 1 / (1 + e^-x)`
- **Rango**: (0, 1)
- **Ventajas**: Interpretable como probabilidad, suave

### 3. **Lineal** (Java - AND)
- **Fórmula**: `f(x) = x`
- **Rango**: (-∞, ∞)
- **Ventajas**: Simple, sin saturación

### 4. **Softmax** (Java - OR)
- **Fórmula**: `softmax(xi) = e^xi / Σ(e^xj)`
- **Rango**: (0, 1), suma = 1
- **Ventajas**: Distribución de probabilidad

### 5. **Escalón** (C# - Riesgo Académico)
- **Fórmula**: `f(x) = 1 si x ≥ 0, sino 0`
- **Rango**: {0, 1}
- **Ventajas**: Salidas binarias claras, perceptrón clásico

### 6. **ReLU** (PHP - Spam)
- **Fórmula**: `ReLU(x) = max(0, x)`
- **Rango**: [0, ∞)
- **Ventajas**: Computacionalmente eficiente, evita gradientes que desaparecen

## 📈 Resultados y Métricas

Cada implementación incluye:
- **Precisión (Accuracy)**
- **Precisión (Precision)**
- **Recall**
- **F1-Score**
- **Matriz de Confusión**
- **Historial de Entrenamiento**

## 🎨 Características Especiales

### Java - Gráficos Swing
- Ventanas interactivas con gráficos en tiempo real
- Visualización del historial de entrenamiento
- Interfaz gráfica nativa

### Python - Gráficos Matplotlib
- Subplots con función de activación
- Guardado automático de imágenes
- Visualización científica

### PHP - Interfaz Web
- Formulario HTML para entrenar el modelo
- Predicción interactiva de emails
- Resultados visuales en navegador
- CSS estilizado

### C# - Estadísticas Detalladas
- Métricas completas de clasificación
- Recomendaciones específicas para estudiantes
- Análisis de riesgo académico

## 🔄 Flujo de Trabajo

1. **Entrenamiento**: Cada perceptrón se entrena con datos sintéticos
2. **Logging**: Se registra el progreso del entrenamiento
3. **Evaluación**: Se calculan métricas de rendimiento
4. **Visualización**: Se generan gráficos (donde aplique)
5. **Guardado**: Se almacenan resultados en archivos .txt
6. **Predicción**: Se prueban datos nuevos

## 📝 Archivos de Salida

- `resultados_clima_entrenamiento.txt`
- `resultados_clima_completos.txt`
- `resultados_fraude_entrenamiento.txt`
- `resultados_fraude_completos.txt`
- `resultados_AND.txt`
- `resultados_AND_completos.txt`
- `resultados_OR.txt`
- `resultados_OR_completos.txt`
- `resultados_riesgo_entrenamiento.txt`
- `resultados_riesgo_completos.txt`
- `resultados_spam_entrenamiento.txt`
- `resultados_spam_completos.txt`
- `grafico_entrenamiento_clima.png`
- `grafico_entrenamiento_fraude.png`

## 🎓 Objetivos Académicos

Este proyecto demuestra:
- Implementación de perceptrones desde cero
- Diferentes funciones de activación
- Aplicaciones de clasificación binaria
- Programación en múltiples lenguajes
- Visualización de datos
- Logging y persistencia
- Interfaces de usuario

## 👥 Casos de Uso del Mundo Real

1. **Predicción del Clima**: Sistemas meteorológicos
2. **Detección de Fraude**: Sistemas bancarios
3. **Funciones Lógicas**: Circuitos digitales
4. **Riesgo Académico**: Sistemas educativos
5. **Detección de Spam**: Filtros de email

---

**Nota**: Todos los datos utilizados son sintéticos y generados para propósitos educativos. Cada implementación incluye más de 10 épocas de entrenamiento como se requiere en las especificaciones.
