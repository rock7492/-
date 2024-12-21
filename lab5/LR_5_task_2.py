import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Завантаження даних
data = np.loadtxt('data_imbalance.txt', delimiter=",")
X, y = data[:, :-1], data[:, -1]

# Перевірка дисбалансу класів
print("Розподіл класів до обробки:", Counter(y))

# 2. Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Використання SMOTE для балансування класів
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Перевірка нового розподілу класів
print("Розподіл класів після SMOTE:", Counter(y_train_balanced))

# 4. Створення і навчання класифікатора
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train_balanced, y_train_balanced)

# 5. Оцінка результатів
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Візуалізація результатів (якщо дані двовимірні)
def visualize_classifier(classifier, X, y):
    """
    Візуалізація меж класифікації для двовимірних даних.
    """
    from matplotlib.colors import ListedColormap

    # Задаємо нові кольори
    background_colors = ['#B3E5FC', '#FFCDD2']  # Світло-блакитний та світло-червоний
    point_colors = ['#0288D1', '#C62828']  # Темно-синій та темно-червоний

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Фон для меж класифікації
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(background_colors))

    # Точки даних
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20,
                cmap=ListedColormap(point_colors))

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid(True)
    plt.show()

# Візуалізація, якщо у даних є лише 2 ознаки
if X_train.shape[1] == 2:
    visualize_classifier(clf, X_test, y_test)
