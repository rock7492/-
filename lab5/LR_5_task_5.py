import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# 1. Завантаження даних із текстового файлу
try:
    data = pd.read_csv('traffic_data.txt', delimiter=",")
except FileNotFoundError:
    raise FileNotFoundError("Файл 'traffic_data.txt' не знайдено. Перевірте шлях до файлу.")

# 2. Перетворення текстових даних у числові
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Збереження енкодера для подальшого декодування

# 3. Відділення ознак і цільової змінної
X = data.iloc[:, :-1]  # Усі стовпці, крім останнього
y = data.iloc[:, -1]   # Останній стовпець — цільова змінна

# Перевірка та балансування класів
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Масштабування ознак (опціонально)
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
# 4. Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# 5. Створення та навчання класифікатора Extra Trees
et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_clf.fit(X_train, y_train)

# 6. Прогнозування та оцінка
y_pred = et_clf.predict(X_test)
print("Класифікаційний звіт:")
print(classification_report(y_test, y_pred, zero_division=1))

# 7. Важливість ознак
feature_importances = et_clf.feature_importances_
feature_importance_data = {
    "Feature": data.columns[:-1],
    "Importance": feature_importances
}
importance_df = pd.DataFrame(feature_importance_data).sort_values(by="Importance", ascending=False)

print("\nВажливість ознак:")
print(importance_df.to_string(index=False))

# 8. Графік важливості ознак
def plot_feature_importance(importance_df):
    """
    Побудова графіка важливості ознак.
    """
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Extra Trees)", fontsize=16)
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue", edgecolor="black")
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().invert_yaxis()  # Перевертає осі для кращого вигляду
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_feature_importance(importance_df)