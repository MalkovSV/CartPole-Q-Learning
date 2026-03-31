import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# ОСНОВНЫЕ НАСТРОЙКИ ОБУЧЕНИЯ
ENVIRONMENT_ID = 'CartPole-v1'  # ID среды Gymnasium для обучения
TARGET_EPISODE_REWARD = 450  # Целевой суммарный reward за эпизод (успешное решение задачи)
MAX_EPISODES = 3000  # Максимальное количество обучающих эпизодов
CONSECUTIVE_SUCCESS_THRESHOLD = 5  # Число последовательных успешных эпизодов для досрочной остановки

# ПАРАМЕТРЫ Q‑ОБУЧЕНИЯ
DISCOUNT_FACTOR_GAMMA = 0.99  # Коэффициент дисконтирования будущих наград (0.9–0.99)

# КОНФИГУРАЦИЯ ДИСКРЕТИЗАЦИИ СОСТОЯНИЙ
STATE_BUCKET_SIZES = [8, 8, 12, 10]  # Число корзин для каждого измерения состояния:
# [позиция тележки, скорость тележки, угол палки, угловая скорость палки]

STATE_VALUE_BOUNDS = [  # Границы значений для каждого измерения (min, max)
    (-4.8, 4.8),   # позиция тележки (м)
    (-3.5, 3.5),   # скорость тележки (м/с)
    (-0.21, 0.21), # угол отклонения палки (рад)
    (-2.0, 2.0)    # угловая скорость палки (рад/с)
]

Q_TABLE_SHAPE = STATE_BUCKET_SIZES + [2]  # Форма Q‑таблицы: [корзины_по_состояниям, число_действий]
q_table = np.zeros(Q_TABLE_SHAPE)
best_q_table = None
best_avg_reward = -np.inf  # Инициализируем минимально возможным значением

def discretize_state(state):
    """
    Преобразует непрерывное состояние в дискретный индекс для Q‑таблицы.

    Args:
        state: непрерывное состояние среды (4 значения)

    Returns:
        tuple: дискретизированный индекс состояния для Q‑таблицы
    """
    discretized = []
    for i in range(len(state)):
        state_i = max(STATE_VALUE_BOUNDS[i][0], min(STATE_VALUE_BOUNDS[i][1], state[i]))
        scale = (state_i - STATE_VALUE_BOUNDS[i][0]) / (STATE_VALUE_BOUNDS[i][1] - STATE_VALUE_BOUNDS[i][0])
        bucket_idx = int(np.floor(scale * STATE_BUCKET_SIZES[i]))
        bucket_idx = min(STATE_BUCKET_SIZES[i] - 1, max(0, bucket_idx))
        discretized.append(bucket_idx)
    return tuple(discretized)

def get_exploration_rate(episode: int) -> float:
    """
    Возвращает текущую вероятность случайного действия (ε) в зависимости от номера эпизода.

    Стратегия:
    - 0–500 эпизодов: максимальное исследование (ε = 1.0)
    - 500–1500 эпизодов: линейное уменьшение до 0.1
    - 1500–3000 эпизодов: плавное снижение до 0.01

    Args:
        episode (int): номер текущего обучающего эпизода

    Returns:
        float: вероятность случайного действия в диапазоне [0.01, 1.0]
    """
    if episode < 500:
        return 1.0
    elif episode < 1500:
        progress = (episode - 500) / 1000
        return 1.0 - (1.0 - 0.1) * progress
    else:
        remaining_episodes = max(0, 3000 - episode)
        return max(0.01, 0.1 * remaining_episodes / 1500)

def get_learning_rate(episode: int) -> float:
    """
    Возвращает текущую скорость обучения (α) в зависимости от номера эпизода.

    Стратегия: поэтапное снижение скорости обучения для стабилизации Q‑значений.

    Args:
        episode (int): номер текущего обучающего эпизода

    Returns:
        float: скорость обучения в диапазоне [0.1, 0.3]
    """
    LEARNING_RATE_PHASES = [
        (0, 1000, 0.3),   # Эпизоды 0–1000: высокая скорость обучения
        (1000, 1500, 0.2), # Эпизоды 1000–1500: умеренная скорость
        (1500, None, 0.1)   # Эпизоды 1500+: низкая скорость для тонкой настройки
    ]
    for start, end, rate in LEARNING_RATE_PHASES:
        if start <= episode and (end is None or episode < end):
            return rate
    return LEARNING_RATE_PHASES[-1][2]  # Возвращаем последнюю фазу, если вышли за пределы

# НАСТРОЙКИ ФОРМИРОВАНИЯ НАГРАДЫ (REWARD SHAPING)
ANGLE_STABILITY_BONUS_WEIGHT = 2.0  # Вес бонуса за вертикальное положение палки
POSITION_CENTERING_BONUS_WEIGHT = 1.0  # Вес бонуса за центрирование тележки

def calculate_reward(state, base_reward):
    """Reward shaping: бонус за стабильность"""
    x, x_dot, theta, theta_dot = state
    # Бонус за угол ближе к вертикали (максимальный при theta = 0)
    angle_bonus = ANGLE_STABILITY_BONUS_WEIGHT * (1 - abs(theta) / 0.21)
    # Бонус за малую скорость тележки (максимальный при x = 0)
    position_bonus = POSITION_CENTERING_BONUS_WEIGHT * (1 - abs(x) / 4.8)
    total_bonus = max(0, angle_bonus + position_bonus)
    return base_reward + total_bonus


# НАСТРОЙКИ СГЛАЖИВАНИЯ И АНАЛИЗА РЕЗУЛЬТАТОВ
EWMA_SMOOTHING_ALPHA = 0.1  # Коэффициент сглаживания для экспоненциального скользящего среднего
REWARD_HISTORY_WINDOW = 50  # Размер окна для расчёта среднего reward (последние N эпизодов)
PROGRESS_REPORT_FREQUENCY = 100  # Частота вывода отчётов о прогрессе (каждые N эпизодов)

def calculate_ewma(scores, alpha=EWMA_SMOOTHING_ALPHA):
    """Экспоненциально взвешенное скользящее среднее"""
    if not scores:
        return 0
    ewma = scores[0]
    for score in scores[1:]:
        ewma = alpha * score + (1 - alpha) * ewma
    return ewma

# КРИТЕРИИ КОНТРОЛЯ КАЧЕСТВА СТРАТЕГИИ
ACTION_DIVERSITY_THRESHOLD = 0.1  # Минимально допустимое соотношение редко/часто выбираемых действий
UNDERPERFORMANCE_MARGIN = 0.8  # Доля от целевого reward для определения «слабой» стратегии

def train_q_learning():
    env = gym.make(ENVIRONMENT_ID)
    scores = []
    avg_scores = []  # Для сглаженного графика
    success_count = 0  # Счётчик последовательных успехов

    best_q_table = None
    best_avg_reward = -np.inf

    for episode in range(MAX_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        discretized_state = discretize_state(state)
        score = 0
        epsilon = get_exploration_rate(episode)
        alpha = get_learning_rate(episode)
        action_counts = [0, 0]  # Счётчик действий 0 и 1

        for t in range(500):  # Максимум 500 шагов в эпизоде
            # Выбор действия: ε‑жадное правило
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Случайное действие
            else:
                action = np.argmax(q_table[discretized_state])  # Лучшее действие из Q‑таблицы

            action_counts[action] += 1
                        # Выполняем действие
            next_state, base_reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True

            # Применяем reward shaping
            reward = calculate_reward(next_state, base_reward)

            # Дискретизируем следующее состояние
            discretized_next_state = discretize_state(next_state)

            # Обновляем Q‑значение согласно алгоритму Q‑learning
            best_next_action = np.argmax(q_table[discretized_next_state])
            td_target = reward + DISCOUNT_FACTOR_GAMMA * q_table[discretized_next_state][best_next_action]
            td_error = td_target - q_table[discretized_state + (action,)]
            q_table[discretized_state + (action,)] += alpha * td_error

            discretized_state = discretized_next_state
            score += base_reward  # Используем оригинальный reward для подсчёта score

            if done:
                break

        scores.append(score)

        # Сглаженный средний reward (последние N эпизодов)
        if len(scores) >= REWARD_HISTORY_WINDOW:
            avg_score = np.mean(scores[-REWARD_HISTORY_WINDOW:])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(np.mean(scores))

        # Контроль за «вырождением» стратегии: проверяем разнообразие действий
        max_actions = max(action_counts)
        if max_actions > 0:
            diversity_ratio = min(action_counts) / max_actions
        else:
            diversity_ratio = 0

        # Если стратегия слишком однообразна и результат слабый — принудительно увеличиваем epsilon
        if (diversity_ratio < ACTION_DIVERSITY_THRESHOLD and
                score < TARGET_EPISODE_REWARD * UNDERPERFORMANCE_MARGIN):
            epsilon = max(epsilon, 0.3)

        # Проверка условия успеха: средний reward за последние N эпизодов выше целевого
        if (len(scores) >= REWARD_HISTORY_WINDOW and
                np.mean(scores[-REWARD_HISTORY_WINDOW:]) >= TARGET_EPISODE_REWARD):
            success_count += 1
            if success_count >= CONSECUTIVE_SUCCESS_THRESHOLD:
                print(f"\nСтабильный успех! Целевой reward {TARGET_EPISODE_REWARD} "
                      f"удерживается {CONSECUTIVE_SUCCESS_THRESHOLD} проверок подряд.")
                break
        else:
            success_count = 0  # Сбрасываем счётчик при провале

        # Сохранение лучшей модели: обновляем, если текущий средний reward выше
        if len(scores) >= REWARD_HISTORY_WINDOW:
            current_avg = np.mean(scores[-REWARD_HISTORY_WINDOW:])
        else:
            current_avg = np.mean(scores[:len(scores)])

        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            best_q_table = q_table.copy()  # Сохраняем копию лучшей таблицы

        # Вывод прогресса с EWMA каждые N эпизодов
        ewma_score = calculate_ewma(scores[-REWARD_HISTORY_WINDOW:])
        if (episode + 1) % PROGRESS_REPORT_FREQUENCY == 0:
            print(f"Эпизод {episode + 1}, EWMA reward: {ewma_score:.2f}, "
                  f"Средний за {REWARD_HISTORY_WINDOW}: {avg_scores[-1]:.2f}")

    env.close()
    return scores, avg_scores, best_q_table, best_avg_reward

if __name__ == "__main__":
    scores, avg_scores, best_model, final_best_avg_reward = train_q_learning()

    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Reward за эпизод', alpha=0.6)
    plt.plot(range(REWARD_HISTORY_WINDOW - 1, len(avg_scores) + REWARD_HISTORY_WINDOW - 1),
              avg_scores, label='Средний reward ({REWARD_HISTORY_WINDOW} эпизодов)', color='red')
    plt.axhline(y=TARGET_EPISODE_REWARD, color='green', linestyle='--',
               label=f'Целевой reward {TARGET_EPISODE_REWARD}')
    plt.xlabel('Эпизод')
    plt.ylabel('Reward')
    plt.title('Обучение Q‑learning на CartPole-v1 (улучшенная версия)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Сохранение лучшей модели
    if best_model is not None:
        np.save('best_q_table.npy', best_model)
        print(f"\nЛучшая модель сохранена в 'best_q_table.npy'")
        print(f"Лучший средний reward: {final_best_avg_reward:.2f}")

    # Статистика обучения
    print(f"\n--- Статистика обучения ---")
    print(f"Всего эпизодов: {len(scores)}")
    print(f"Максимальный reward за эпизод: {max(scores):.2f}")
    print(f"Средний reward за все эпизоды: {np.mean(scores):.2f}")
    if len(scores) >= 100:
        print(f"Средний reward за последние 100 эпизодов: {np.mean(scores[-100:]):.2f}")


    achieved_target = final_best_avg_reward >= TARGET_EPISODE_REWARD
    print(f"Достигнут целевой reward {TARGET_EPISODE_REWARD}: {'Да' if achieved_target else 'Нет'}")

    # Тестирование лучшей модели
    print(f"\nТестирование лучшей модели...")
    test_episodes = 10
    test_env = gym.make(ENVIRONMENT_ID)
    test_scores = []

    for episode in range(test_episodes):
        state = test_env.reset()
        if isinstance(state, tuple):
            state = state[0]
        discretized_state = discretize_state(state)
        score = 0

        for t in range(500):
            action = np.argmax(best_model[discretized_state])
            next_state, reward, done, truncated, _ = test_env.step(action)
            if truncated:
                done = True
            discretized_state = discretize_state(next_state)
            score += reward
            if done:
                break
        test_scores.append(score)

    test_env.close()
    print(f"Результаты тестирования лучшей модели ({test_episodes} эпизодов):")
    print(f"Средний reward: {np.mean(test_scores):.2f}")
    print(f"Минимальный reward: {min(test_scores)}")
    print(f"Максимальный reward: {max(test_scores)}")
    successful_tests = sum(1 for s in test_scores if s >= TARGET_EPISODE_REWARD)
    print(f"Успешных эпизодов (reward ≥ {TARGET_EPISODE_REWARD}): "
          f"{successful_tests} из {test_episodes}")
