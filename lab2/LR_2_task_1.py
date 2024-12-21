import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier

input_file = 'income_data.txt'

X = []
y = []
max_samples_per_class = 25000
class1_count, class2_count = 0, 0

with open(input_file, 'r') as file:
    for line in file:
        if '?' in line:
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
        if class1_count >= max_samples_per_class and class2_count >= max_samples_per_class:
            break

X = np.array(X)
y = np.array(y)

encoders = []
X_encoded = np.empty_like(X, dtype=int)
for i in range(X.shape[1]):
    column = X[:, i]
    if not np.char.isnumeric(column).all():
        encoder = LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(column)
        encoders.append(encoder)
    else:
        X_encoded[:, i] = column.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

classifier = OneVsOneClassifier(LinearSVC(random_state=42))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

new_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
            'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

new_data_encoded = []

for i, item in enumerate(new_data):
    if item.isdigit():
        new_data_encoded.append(int(item))
    else:
        if i < len(encoders):  # Перевірка, чи є енкодер для цієї категорії
            try:
                encoded_value = encoders[i].transform([item])[0]
                new_data_encoded.append(encoded_value)
            except ValueError:  # Якщо значення не знайдено в енкодері
                encoders[i].fit([item])  # Перенавчаємо енкодер для додавання нового значення
                encoded_value = encoders[i].transform([item])[0]
                new_data_encoded.append(encoded_value)
        else:
            new_data_encoded.append(-1)  # Якщо для цієї категорії немає енкодера, присвоюємо -1

new_data_encoded = np.array(new_data_encoded)

predicted_class = classifier.predict([new_data_encoded])
predicted_income = '<=50K' if predicted_class == 0 else '>50K'

print(f"Прогнозований дохід для нового зразка: {predicted_income}")
