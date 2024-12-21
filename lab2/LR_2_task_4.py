import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Спроба завантажити файл
input_file = 'income_data.txt'

try:
    # Спроба відкрити файл
    with open(input_file, 'r') as file:
        X = []
        y = []
        max_samples_per_class = 5000  # Зменшено для швидшого тестування
        class1_count, class2_count = 0, 0
        for line in file:
            if '?' in line:  # Пропуск рядків із відсутніми даними
                continue
            data = line.strip().split(', ')
            if data[-1] == '<=50K' and class1_count < max_samples_per_class:
                X.append(data[:-1])
                y.append(0)
                class1_count += 1
            elif data[-1] == '>50K' and class2_count < max_samples_per_class:
                X.append(data[:-1])
                y.append(1)
                class2_count += 1
except FileNotFoundError:
    # Якщо файл не знайдено, створюємо тестові дані
    print("Файл не знайдено. Використовуються тестові дані.")
    X = [
        ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'],
        ['50', 'Self-emp-not-inc', '83311', 'Bachelors', '13', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', '0', '0', '13', 'United-States']
    ]
    y = [0, 1]

# Перетворення на NumPy масиви
X = np.array(X)
y = np.array(y)

# Кодування текстових ознак
label_encoders = []
X_encoded = np.empty_like(X, dtype=int)
for i in range(X.shape[1]):
    column = X[:, i]
    if not np.char.isnumeric(column).all():
        le = LabelEncoder()
        X_encoded[:, i] = le.fit_transform(column)
        label_encoders.append(le)
    else:
        X_encoded[:, i] = column.astype(int)

X = X_encoded

# Розділення на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список моделей для класифікації
models = [
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Support Vector Machine', SVC(kernel='rbf', gamma='auto'))
]

# Функція для оцінки моделі
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return name, accuracy, precision, recall, f1

# Оцінка кожної моделі
results = []
for name, model in models:
    metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(metrics)

# Результати в DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df.set_index('Model', inplace=True)

# Виведення результатів
print("\nРезультати порівняння моделей:")
print(results_df)

# Візуалізація результатів
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Порівняння моделей за метриками якості")
plt.ylabel("Значення метрики")
plt.xticks(rotation=45)
plt.show()

# Вибір найкращої моделі
best_model = results_df['F1 Score'].idxmax()
print(f"\nНайкраща модель за F1 Score: {best_model}")
