import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.8, 1.6, 1.9])
# Побудова матриці Вандермонда для полінома 4-го ступеня
X = np.vander(x, increasing=True)

# Розрахунок коефіцієнтів полінома
coefficients = np.linalg.solve(X, y)

# Функція полінома на основі знайдених коефіцієнтів
def polynomial(x_value):
    return sum(c * x_value**i for i, c in enumerate(coefficients))

# Генерація точок для графіка
x_vals = np.linspace(0.1, 0.7, 500)
y_vals = [polynomial(val) for val in x_vals]

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Інтерполяційний поліном", color="blue")
plt.scatter(x, y, color="red", label="Задані точки")
plt.title("Інтерполяція поліномом 4-го ступеня")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Обчислення значень у точках 0.2 і 0.5
y_at_0_2 = polynomial(0.2)
y_at_0_5 = polynomial(0.5)

# Виведення результатів
print(f"Значення функції у точці x=0.2: {y_at_0_2:.3f}")
print(f"Значення функції у точці x=0.5: {y_at_0_5:.3f}")