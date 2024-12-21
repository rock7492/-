import pandas as pd
import matplotlib.pyplot as plt

# Вхідні дані
data = [
    ['Sunny', 'High', 'Weak', 'No'],
    ['Sunny', 'High', 'Strong', 'No'],
    ['Overcast', 'High', 'Weak', 'Yes'],
    ['Rain', 'High', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'High', 'Weak', 'No'],
    ['Sunny', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'High', 'Strong', 'No']
]

# Перетворення в DataFrame
df = pd.DataFrame(data, columns=['Outlook', 'Humidity', 'Wind', 'Play'])

# Розрахунок ймовірностей
def calculate_probability(condition, target, df):
    condition_prob = len(df[(df[condition[0]] == condition[1]) & (df['Play'] == target)]) / len(df[df['Play'] == target])
    return condition_prob

# Умови
conditions = [
    ('Outlook', 'Rain'),
    ('Humidity', 'High'),
    ('Wind', 'Weak')
]

# Розрахунок ймовірностей для "Yes" і "No"
p_yes = len(df[df['Play'] == 'Yes']) / len(df)
p_no = len(df[df['Play'] == 'No']) / len(df)

p_yes_given_conditions = p_yes
p_no_given_conditions = p_no

for condition in conditions:
    p_yes_given_conditions *= calculate_probability(condition, 'Yes', df)
    p_no_given_conditions *= calculate_probability(condition, 'No', df)

# Нормалізація
total = p_yes_given_conditions + p_no_given_conditions
p_yes_final = p_yes_given_conditions / total
p_no_final = p_no_given_conditions / total

# Результат
print(f"Ймовірність 'Yes' (відбудеться матч): {p_yes_final:.2f}")
print(f"Ймовірність 'No' (матч не відбудеться): {p_no_final:.2f}")

if p_yes_final > p_no_final:
    print("Результат: матч відбудеться.")
else:
    print("Результат: матч не відбудеться.")

# Візуалізація ймовірностей
labels = ['Yes (Матч відбудеться)', 'No (Матч не відбудеться)']
probabilities = [p_yes_final, p_no_final]

plt.figure(figsize=(8, 6))
plt.bar(labels, probabilities, color=['green', 'red'], alpha=0.7)
plt.title('Ймовірність проведення матчу', fontsize=14)
plt.ylabel('Ймовірність', fontsize=12)
plt.ylim(0, 1)  # Ймовірності від 0 до 1
for i, prob in enumerate(probabilities):
    plt.text(i, prob + 0.02, f"{prob:.2f}", ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()