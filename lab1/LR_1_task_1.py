import numpy as np
import matplotlib.pyplot as plt

# Функція активації
def step_function(x):
    return 1 if x >= 0 else 0

# Персептрон для функції OR
def or_perceptron(x1, x2):
    weights = np.array([1, 1])  # Ваги
    threshold = -0.5  # Поріг
    inputs = np.array([x1, x2])  # Вхідні дані
    linear_combination = np.dot(weights, inputs) + threshold  # Лінійна комбінація
    return step_function(linear_combination)

# Персептрон для функції AND
def and_perceptron(x1, x2):
    weights = np.array([1, 1])  # Ваги
    threshold = -1.5  # Поріг
    inputs = np.array([x1, x2])  # Вхідні дані
    linear_combination = np.dot(weights, inputs) + threshold  # Лінійна комбінація
    return step_function(linear_combination)

# Персептрон для функції XOR через OR та AND
def xor_perceptron(x1, x2):
    or_result = or_perceptron(x1, x2)
    and_result = and_perceptron(x1, x2)
    return step_function(or_result - and_result)  # XOR через OR та NOT AND

# Генерація випадкових точок
np.random.seed(42)  # Для відтворюваності результатів
num_points = 200  # Кількість точок
x_random = np.random.rand(num_points) * 2 - 0.5  # Генерація значень від -0.5 до 1.5
y_random = np.random.rand(num_points) * 2 - 0.5  # Генерація значень від -0.5 до 1.5

# Класифікація точок за допомогою XOR
xor_results = np.array([xor_perceptron(x, y) for x, y in zip(x_random, y_random)])

# Встановлення кольорів: синій для 1, оранжевий для 0
colors_xor = ['blue' if result == 1 else 'orange' for result in xor_results]

# Побудова графіку
plt.figure(figsize=(8, 6))
plt.scatter(x_random, y_random, c=colors_xor, alpha=0.7)
plt.title('Функція XOR')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()