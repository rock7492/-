import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Експериментальні дані
X = np.array([0, 5, 10, 15, 20, 25])
Y = np.array([21, 39, 51, 63, 70, 90])

# Визначення лінійної функції для апроксимації
def linear_func(x, a, b):
    return a * x + b

# Знаходження параметрів функції методом найменших квадратів
params, _ = curve_fit(linear_func, X, Y)
a, b = params

# Генерація значень для побудови апроксимуючої функції
X_fit = np.linspace(min(X), max(X), 500)
Y_fit = linear_func(X_fit, a, b)

# Побудова графіку
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='red', label='Експериментальні дані')
plt.plot(X_fit, Y_fit, color='blue', label=f'Апроксимуюча функція: Y = {a:.2f}X + {b:.2f}')
plt.title('Апроксимація методом найменших квадратів')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()