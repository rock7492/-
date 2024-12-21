import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Генерація випадкових даних
m = 100   # Кількість точок (зразків)
X = np.linspace(-3, 3, m).reshape(-1, 1)   # Перетворюємо X у 2D масив (очікуваний формат для sklearn)
y = 2 * np.sin(X).ravel() + np.random.uniform(-0.6, 0.6, m)   # y залишається як одномірний масив

# Створення поліноміальних ознак для X
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Виведення деяких значень для перевірки
print("X[0] =", X[0])
print("X_poly[0] =", X_poly[0])

# Лінійна регресія на поліноміальних ознаках
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)  # Навчаємо модель

# Отримуємо коефіцієнти та інтерсепт
intercept = lin_reg.intercept_
coef = lin_reg.coef_
print("Інтерсепт (Intercept):", intercept)
print("Коефіцієнти (coef):", coef)

# Прогнозування значень на нових даних
x_new = np.linspace(min(X), max(X), 100).reshape(-1, 1)  # Нові дані для прогнозу
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Дані', alpha=0.7)
plt.plot(x_new, y_new, color='red', label='Поліноміальна регресія (степінь 2)', linewidth=2)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Поліноміальна регресія (степінь 2)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()