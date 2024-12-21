import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Завантаження даних
try:
    data = np.loadtxt('data_random_forests.txt', delimiter=",")
except FileNotFoundError:
    raise FileNotFoundError("Файл 'data_random_forests.txt' не знайдено. Переконайтеся, що файл існує.")

# Розбиття даних на ознаки (X) та мітки (y)
X, y = data[:, :-1], data[:, -1]

# 2. Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Визначення параметрів для сіткового пошуку
param_grid = {
    'n_estimators': [50, 100, 200],  # Кількість дерев у лісі
    'max_depth': [None, 10, 20, 30],  # Максимальна глибина дерева
    'min_samples_split': [2, 5, 10],  # Мінімальна кількість зразків для поділу вузла
    'min_samples_leaf': [1, 2, 4],  # Мінімальна кількість зразків у листі
    'bootstrap': [True, False]  # Використання підвибірки (Bootstrap)
}

# 4. Ініціалізація класифікатора
rf_clf = RandomForestClassifier(random_state=42)

# 5. Сітковий пошук із перехресною перевіркою
grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,  # Кількість фолдів для крос-валідації
    scoring='accuracy',  # Метрика для оцінки
    verbose=2,  # Рівень деталізації логів
    n_jobs=-1  # Використання всіх процесорів
)

# Навчання на тренувальних даних
grid_search.fit(X_train, y_train)

# 6. Виведення результатів
print("\nНайкращі параметри моделі:")
print(grid_search.best_params_)

print("\nНайкраща точність на тренувальних даних:")
print(f"{grid_search.best_score_:.4f}")

# Оцінка моделі з найкращими параметрами на тестових даних
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report на тестових даних:")
print(classification_report(y_test, y_pred))
