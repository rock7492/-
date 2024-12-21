import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Обмежуємо кількість даних для швидшого тестування
max_datapoints = 5000  # Зменшено для швидшого виконання
# 1. Завантаження та обробка даних
input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        if not line.strip():  # Пропуск порожніх рядків
            continue
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:  # Пропуск рядків із відсутніми даними
            continue
        data = line.strip().split(', ')
        if len(data) != 15:  # Перевірка структури рядка
            continue
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

# 2. Кодування текстових даних
X = np.array(X)
y = np.array(y)
label_encoders = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)
X = X_encoded.astype(int)

# 3. Розбиття даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функція для обчислення метрик якості
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# 4. Моделі з різними ядрами

# Поліноміальне ядро
poly_clf = SVC(kernel='poly', degree=3)  # Використовуємо 3-й ступінь для швидкості
poly_clf.fit(X_train, y_train)
poly_metrics = evaluate_model(poly_clf, X_test, y_test)
# Гаусівське (RBF) ядро
rbf_clf = SVC(kernel='rbf')
rbf_clf.fit(X_train, y_train)
rbf_metrics = evaluate_model(rbf_clf, X_test, y_test)

# Сигмоїдальне ядро
sigmoid_clf = SVC(kernel='sigmoid')
sigmoid_clf.fit(X_train, y_train)
sigmoid_metrics = evaluate_model(sigmoid_clf, X_test, y_test)

# 5. Порівняння результатів

print("\nПорівняння якості моделей SVM з різними ядрами:")
print(f"Поліноміальне ядро:")
print(f"  Акуратність: {poly_metrics[0]:.2f}, Точність: {poly_metrics[1]:.2f}, Повнота: {poly_metrics[2]:.2f}, F1-міра: {poly_metrics[3]:.2f}")
print(f"Гаусівське ядро (RBF):")
print(f"  Акуратність: {rbf_metrics[0]:.2f}, Точність: {rbf_metrics[1]:.2f}, Повнота: {rbf_metrics[2]:.2f}, F1-міра: {rbf_metrics[3]:.2f}")
print(f"Сигмоїдальне ядро:")
print(f"  Акуратність: {sigmoid_metrics[0]:.2f}, Точність: {sigmoid_metrics[1]:.2f}, Повнота: {sigmoid_metrics[2]:.2f}, F1-міра: {sigmoid_metrics[3]:.2f}")

# Додаткове порівняння результатів для кожного ядра
def summarize_results(kernel_name, metrics):
    print(f"\n{kernel_name} модель:")
    print(f"  - Акуратність (Accuracy): {metrics[0]:.2f}")
    print(f"  - Точність (Precision): {metrics[1]:.2f}")
    print(f"  - Повнота (Recall): {metrics[2]:.2f}")
    print(f"  - F1-міра (F1 Score): {metrics[3]:.2f}")

summarize_results("Поліноміальне", poly_metrics)
summarize_results("Гаусівське (RBF)", rbf_metrics)
summarize_results("Сигмоїдальне", sigmoid_metrics)
