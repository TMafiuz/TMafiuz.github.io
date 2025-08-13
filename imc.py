import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.linear_model import LinearRegression

# =======================
# 1. ENTRENAR MODELO
# =======================
np.random.seed(42)
pesos = np.random.uniform(50, 100, 10)  # 50 a 100 kg
alturas = np.random.uniform(1.5, 1.8, 10)  # 1.50 a 1.80 m
imcs = pesos / (alturas ** 2)

X = np.column_stack((pesos, alturas))
y = imcs

modelo = LinearRegression()
modelo.fit(X, y)

# =======================
# 2. FUNCIONES
# =======================
def predecir_imc():
    try:
        peso = float(entry_peso.get())
        altura = float(entry_altura.get())
        if altura <= 0:
            raise ValueError("La altura debe ser mayor que cero.")
        imc_pred = modelo.predict([[peso, altura]])[0]
        messagebox.showinfo("Resultado", f"IMC estimado: {imc_pred:.2f}")
    except ValueError:
        messagebox.showerror("Error", "Introduce valores numéricos válidos.")

# =======================
# 3. INTERFAZ TKINTER
# =======================
root = tk.Tk()
root.title("Calculadora de IMC con IA")
root.geometry("300x200")

label_titulo = tk.Label(root, text="Calculadora de IMC (Regresión)", font=("Arial", 12, "bold"))
label_titulo.pack(pady=10)

frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=5)

label_peso = tk.Label(frame_inputs, text="Peso (kg):")
label_peso.grid(row=0, column=0, padx=5, pady=5)
entry_peso = tk.Entry(frame_inputs)
entry_peso.grid(row=0, column=1, padx=5, pady=5)

label_altura = tk.Label(frame_inputs, text="Altura (m):")
label_altura.grid(row=1, column=0, padx=5, pady=5)
entry_altura = tk.Entry(frame_inputs)
entry_altura.grid(row=1, column=1, padx=5, pady=5)

btn_predecir = tk.Button(root, text="Calcular IMC", command=predecir_imc)
btn_predecir.pack(pady=10)

root.mainloop()
