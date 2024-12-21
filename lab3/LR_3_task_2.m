% Основний код для визначення режиму роботи кондиціонера на основі температури та її зміни

% Початкові значення температури та швидкості її зміни
temperature = 18; % Температура в приміщенні (°C)
temp_change_rate = -1; % Швидкість зміни температури (°C/хв)

% Ініціалізація змінних для режиму роботи кондиціонера та часу
ac_state = zeros(1, 11); % Збереження режимів для побудови графіка (11 точок часу)
time_points = 0:10; % Час для побудови графіка (від 0 до 10 хв)

% Цикл для моделювання змін у режимі роботи кондиціонера
for t = 1:length(time_points)
    % Правила керування залежно від температури та швидкості її зміни
    if temperature > 30 && temp_change_rate > 0
        ac_state(t) = 70; % Великий кут вліво для охолодження
    elseif temperature > 30 && temp_change_rate < 0
        ac_state(t) = 40; % Малий кут вліво для охолодження
    elseif temperature > 20 && temp_change_rate > 0
        ac_state(t) = 70; % Великий кут вліво для охолодження
    elseif temperature > 20 && temp_change_rate < 0
        ac_state(t) = 10; % Вимкнути кондиціонер
    elseif temperature < 15 && temp_change_rate < 0
        ac_state(t) = 90; % Великий кут вправо для обігріву
    elseif temperature < 15 && temp_change_rate > 0
        ac_state(t) = 50; % Малий кут вправо для обігріву
    elseif temperature < 20 && temp_change_rate < 0
        ac_state(t) = 70; % Великий кут вліво для охолодження
    elseif temperature < 20 && temp_change_rate > 0
        ac_state(t) = 10; % Вимкнути кондиціонер
    elseif temperature == 0 && temp_change_rate == 0
        ac_state(t) = 10; % Вимкнути кондиціонер
    else
        ac_state(t) = 10; % За замовчуванням: вимкнути кондиціонер
    end

    % Оновлення температури та швидкості її зміни
    temperature = temperature + temp_change_rate; % Зміна температури
end

% Побудова графіка режиму роботи кондиціонера у часі
figure;
plot(time_points, ac_state, 'LineWidth', 2, 'Marker', 's', 'Color', 'b'); % Графік із синім кольором і маркерами
xlabel('Час (хвилини)');
ylabel('Режим кондиціонера');
title('Режим роботи кондиціонера у часі');
grid on;

% Виведення результату для останнього моменту часу
fprintf('Остаточний режим роботи кондиціонера: %.2f\n', ac_state(end));
