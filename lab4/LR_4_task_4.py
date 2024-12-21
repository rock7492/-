import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження набору даних для діабету
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбиття на тренувальні та тестові набори
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення лінійної моделі регресії
regr = linear_model.LinearRegression()
regr.fit(Xtrain, Ytrain)

# Прогнозування значень на тестових даних
ypred = regr.predict(Xtest)

# Виведення коефіцієнтів та інтерсепту моделі
print(f"Коефіцієнт регресії (regr.coef_): {regr.coef_}")
print(f"Інтерсепт (regr.intercept_): {regr.intercept_}")

# Оцінка моделі
print(f"R^2 (r2_score): {r2_score(Ytest, ypred)}")
print(f"Середня абсолютна похибка (mean_absolute_error): {mean_absolute_error(Ytest, ypred)}")
print(f"Середньоквадратична похибка (mean_squared_error): {mean_squared_error(Ytest, ypred)}")

# Візуалізація результатів
fig, ax = plt.subplots(figsize=(8, 6))  # Збільшений розмір графіка

# Графік порівняння реальних і передбачених значень
ax.scatter(Ytest, ypred, color='b', edgecolors='black', alpha=0.7, label='Прогнозовані значення')

# Додавання ідеальної лінії (y = x) для порівняння
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label='Ідеальна лі-нія')

# Налаштування графіка
ax.set_xlabel('Виміряно (реальні значення)', fontsize=12)
ax.set_ylabel('Передбачено (модель)', fontsize=12)
ax.set_title('Порівняння реальних і передбачених значень', fontsize=14)
ax.legend(loc='best', fontsize=12)
ax.grid(True)

# Показ графіка
plt.show()