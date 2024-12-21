# ===================================================
# Приклад класифікатора Ridge
# ======================================================================
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ініціалізація та тренування RidgeClassifier
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Метрики
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Precision:', np.round(metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0), 4))
print('Recall:', np.round(metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))
print('\t\tClassification Report:\n', metrics.classification_report(y_test, y_pred))

# Матриця плутанини
mat = confusion_matrix(y_test, y_pred)
sns.set()  # Стиль для графіка
plt.figure(figsize=(8, 6))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title("Confusion Matrix")
plt.savefig("Confusion.jpg")

# Збереження графіка у SVG
f = BytesIO()
plt.savefig(f, format="svg")
plt.show()
