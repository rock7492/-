import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

input_file = 'data_singlevar_regr.txt'

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

# Створення об'єкта для лінійної регресії
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування результату
y_test_pred = regressor.predict(X_test)

# Побудова графіка з іншими стилями та кольорами
plt.figure(figsize=(8, 6))

# Зміна кольору точок (реальних даних) на синій
plt.scatter(X_test, y_test, color='blue', label='Реальні дані', alpha=0.6)  # Сині точки для реальних даних

# Зміна кольору лінії прогнозу на червоний
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Прогнозована лінія')  # Червона лінія для прогнозів

# Додавання заголовка, міток осей, легенди та сітки
plt.title('Лінійна регресія: тест проти прогнозу', fontsize=14)
plt.xlabel('Вхідна змінна')
plt.ylabel('Цільова змінна')
plt.legend()
plt.grid(True)
# Показати графік
plt.show()

# Збереження моделі у файл за допомогою pickle
with open('linear_regressor_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Оцінка моделі за допомогою різних метрик
print("Результати роботи лінійної регресії:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Корінь середньоквадратичної похибки =", round(np.sqrt(sm.mean_squared_error(y_test, y_test_pred)), 2))  # RMSE
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Коефіцієнт поясненої дисперсії =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 =", round(sm.r2_score(y_test, y_test_pred), 2))

# Завантаження та перевірка моделі, якщо це потрібно
with open('linear_regressor_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Прогнозування за допомогою завантаженої моделі (перевірка)
y_test_pred_loaded_model = loaded_model.predict(X_test)
print("R2 для завантаженої моделі:", round(sm.r2_score(y_test, y_test_pred_loaded_model), 2))