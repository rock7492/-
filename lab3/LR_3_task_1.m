% Нечіткі функції належності для температури та тиску
x_range = linspace(-100, 100, 300); % Розширений діапазон значень для x (температура, тиск, кут)

% Трапецієподібна функція належності
def_trapmf = @(x, a, b, c, d) max(min((x - a) / (b - a), (d - x) / (d - c)), 0);
% Трикутна функція належності
def_trimf = @(x, a, b, c) max(min((x - a) / (b - a), (c - x) / (c - b)), 0);

% Функції належності для температури
temp_cold = def_trapmf(x_range, -15, -5, 10, 25);
temp_warm = def_trimf(x_range, 15, 35, 55);
temp_hot = def_trapmf(x_range, 50, 70, 90, 110);

% Функції належності для тиску
pressure_weak = def_trapmf(x_range, -2, 0, 2, 4);
pressure_normal = def_trimf(x_range, 3, 6, 9);
pressure_strong = def_trapmf(x_range, 8, 10, 12, 14);

% Тестові значення
test_temperature = 65;  % Приклад температури
test_pressure = 6;      % Приклад тиску

% Розрахунок значень належності для температури та тиску
membership_temp_cold = def_trapmf(test_temperature, -15, -5, 10, 25);
membership_temp_warm = def_trimf(test_temperature, 15, 35, 55);
membership_temp_hot = def_trapmf(test_temperature, 50, 70, 90, 110);

membership_pressure_weak = def_trapmf(test_pressure, -2, 0, 2, 4);
membership_pressure_normal = def_trimf(test_pressure, 3, 6, 9);
membership_pressure_strong = def_trapmf(test_pressure, 8, 10, 12, 14);

% Виведення значень належності для тестових значень
disp("Належність температури:");
disp(["Холодна вода: ", num2str(membership_temp_cold)]);
disp(["Тепла вода: ", num2str(membership_temp_warm)]);
disp(["Гаряча вода: ", num2str(membership_temp_hot)]);

disp("Належність тиску:");
disp(["Слабкий тиск: ", num2str(membership_pressure_weak)]);
disp(["Нормальний тиск: ", num2str(membership_pressure_normal)]);
disp(["Сильний тиск: ", num2str(membership_pressure_strong)]);

% Графіки функцій належності
subplot(2, 2, 1);
plot(x_range, temp_cold, 'r--', x_range, temp_warm, 'g--', x_range, temp_hot, 'b--');
title('Функції належності температури');
legend('Холодна', 'Тепла', 'Гаряча');
xlabel('Температура');
ylabel('Рівень належності');

title('Функції належності тиску');
subplot(2, 2, 2);
plot(x_range, pressure_weak, 'm-.', x_range, pressure_normal, 'c-.', x_range, pressure_strong, 'k-.');
title('Функції належності тиску');
legend('Слабкий', 'Нормальний', 'Сильний');
xlabel('Тиск');
ylabel('Рівень належності');

% Оцінка правил
hot_valve_angle = 0;
cold_valve_angle = 0;

% Правила для гарячої води
hot_valve_angle = max(hot_valve_angle, min(membership_temp_cold, membership_pressure_strong) * -40);  % Змінене правило 1
hot_valve_angle = max(hot_valve_angle, min(membership_temp_hot, membership_pressure_normal) * -50);  % Змінене правило 2
hot_valve_angle = max(hot_valve_angle, min(membership_temp_warm, membership_pressure_strong) * -25);  % Змінене правило 3
hot_valve_angle = max(hot_valve_angle, min(membership_temp_warm, membership_pressure_weak) * 20);    % Змінене правило 4

% Правила для холодної води
cold_valve_angle = max(cold_valve_angle, min(membership_temp_cold, membership_pressure_strong) * 50); % Змінене правило 5
cold_valve_angle = max(cold_valve_angle, min(membership_temp_warm, membership_pressure_strong) * -10); % Змінене правило 6
cold_valve_angle = max(cold_valve_angle, min(membership_temp_hot, membership_pressure_weak) * 25);   % Змінене правило 7
cold_valve_angle = max(cold_valve_angle, min(membership_temp_hot, membership_pressure_strong) * -35); % Змінене правило 8

% Виведення кінцевих результатів
disp('Кут відкриття гарячого крану:');
disp(hot_valve_angle);
disp('Кут відкриття холодного крану:');
disp(cold_valve_angle);