import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

input_file = 'data_multivar_regr.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальну та тестову вибірки
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Навчальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створення лінійного регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування результату для лінійної регресії
y_test_pred = regressor.predict(X_test)
# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

# Створення моделі для поліноміальної регресії
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Прогнозування для поліноміальної регресії
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

# Виведення результатів для лінійної та поліноміальної регресії
print("\nПрогнозування для лінійної регресії:\n", regressor.predict([[7.75, 6.35, 5.56]]))
print("\nПрогнозування для поліноміальної регресії:\n", poly_linear_model.predict(polynomial.fit_transform([[7.75, 6.35, 5.56]])))

# Оцінка моделей
print("\nМетрики ефективності лінійної регресії:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 =", round(sm.r2_score(y_test, y_test_pred), 2))

print("\nМетрики ефективності поліноміальної регресії:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("R2 =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

# Візуалізація результатів для порівняння моделей
plt.figure(figsize=(10, 6))

# Графік для лінійної регресії
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, color='blue', label='Прогнози лінійної регресії')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ідеальна лінія')
plt.title('Лінійна регресія')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.legend()
plt.grid(True)

# Графік для поліноміальної регресії
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_poly, color='green', label='Прогнози поліноміальної регресії')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ідеальна лінія')
plt.title('Поліноміальна регресія')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()