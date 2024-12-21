import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

# 1. Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Перевірка структури даних
print("Перші 5 рядків датасету:")
print(dataset.head())
print("\nРозмір датасету:", dataset.shape)

# Розділення ознак і класів
X = dataset.iloc[:, :-1].values  # Ознаки (довжина і ширина чашолистків і пелюсток)
y = dataset.iloc[:, -1].values   # Класи (сорт ірису)

# 2. Розділення даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Визначення моделей для класифікації
models = [
    ('Logistic Regression', OneVsRestClassifier(LogisticRegression(solver='liblinear'))),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Support Vector Machine', SVC(kernel='rbf', gamma='auto'))
]

# 4. Функція для оцінки кожної моделі
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name}: Акуратність ={accuracy:.2f}, Точність={precision:.2f}, Повнота={recall:.2f}, F1-міра={f1:.2f}")
    return accuracy, precision, recall, f1

# 5. Оцінка та порівняння моделей
results = []
for name, model in models:
    print(f"\nОцінка моделі: {name}")
    metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append((name, *metrics))

# 6. Порівняння моделей за метриками
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df.set_index('Model', inplace=True)

# 7. Візуалізація порівняння моделей
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Порівняння моделей за метриками якості")
plt.ylabel("Значення метрики")
plt.xticks(rotation=45)
plt.show()

# 8. Підсумковий звіт з найкращою моделлю
print("\nРезультати порівняння моделей:")
print(results_df)

# Отримуємо найкращу модель за F1 Score
best_model = results_df['F1 Score'].idxmax()
print(f"\nНайкраща модель за F1 Score: {best_model}")
