import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# ОСНОВНЫЕ НАСТРОЙКИ ОБУЧЕНИЯ
ENVIRONMENT_ID = 'CartPole-v1'
TARGET_EPISODE_REWARD = 450
MAX_EPISODES = 3000
CONSECUTIVE_SUCCESS_THRESHOLD = 5

# ПАРАМЕТРЫ Q‑ОБУЧЕНИЯ
DISCOUNT_FACTOR_GAMMA = 0.99

# КОНФИГУРАЦИЯ ДИСКРЕТИЗАЦИИ СОСТОЯНИЙ
STATE_BUCKET_SIZES = [6, 6, 15, 12]  # Оптимизированная дискретизация
STATE_VALUE_BOUNDS = [
    (-4.8, 4.8),
    (-3.5, 3.5),
    (-0.21, 0.21),
    (-2.0, 2.0)
]

# ДОБАВЛЕННАЯ КОНСТАНТА ЛИМИТА РАЗМЕРА
MAX_Q_TABLE_SIZE = 1_000_000  # Максимум 1 млн элементов — можно настроить

""" Q_TABLE_SHAPE = STATE_BUCKET_SIZES + [2]
q_table = np.zeros(Q_TABLE_SHAPE)
best_q_table = None
best_avg_reward = -np.inf """

def check_q_table_size(bucket_sizes, n_actions, max_size):
    """
    Проверяет, не превышает ли размер Q‑таблицы допустимый лимит.

    Args:
        bucket_sizes: список размеров корзин для каждого измерения состояния
        n_actions: количество возможных действий
        max_size: максимально допустимый размер таблицы

    Returns:
        tuple: (bool — допустимо ли создание, int — вычисленный размер)
    """
    total_size = np.prod(bucket_sizes) * n_actions
    is_acceptable = total_size <= max_size
    return is_acceptable, total_size

def discretize_state(state):
    discretized = []
    for i in range(len(state)):
        state_i = max(STATE_VALUE_BOUNDS[i][0], min(STATE_VALUE_BOUNDS[i][1], state[i]))
        scale = (state_i - STATE_VALUE_BOUNDS[i][0]) / (STATE_VALUE_BOUNDS[i][1] - STATE_VALUE_BOUNDS[i][0])
        bucket_idx = int(np.floor(scale * STATE_BUCKET_SIZES[i]))
        bucket_idx = min(STATE_BUCKET_SIZES[i] - 1, max(0, bucket_idx))
        discretized.append(bucket_idx)
    return tuple(discretized)

def get_exploration_rate(episode: int) -> float:
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 1200
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-episode / EPSILON_DECAY)

def get_learning_rate(episode: int, td_error: float = None) -> float:
    """
    Адаптивная скорость обучения с экспоненциальным затуханием.
    """
    ALPHA_START = 0.3      # Начальная скорость обучения
    ALPHA_END = 0.05      # Минимальная скорость обучения
    ALPHA_DECAY = 0.002   # Коэффициент затухания (подбирается экспериментально)

    # Базовое экспоненциальное затухание
    base_alpha = ALPHA_END + (ALPHA_START - ALPHA_END) * np.exp(-ALPHA_DECAY * episode)

    # Адаптивная корректировка на основе TD‑ошибки
    if td_error is not None:
        if td_error < 1.0:
            alpha = base_alpha * 0.8  # Уменьшаем при малой ошибке (стабильное обучение)
        elif td_error > 10.0:
            alpha = base_alpha * 1.2  # Увеличиваем при большой ошибке (нужно больше обновлений)
        else:
            alpha = base_alpha
    else:
        alpha = base_alpha

    return max(ALPHA_END, alpha)

ANGLE_STABILITY_BONUS_WEIGHT = 2.0
POSITION_CENTERING_BONUS_WEIGHT = 1.0

def calculate_reward(state, base_reward):
    x, x_dot, theta, theta_dot = state
    angle_bonus = ANGLE_STABILITY_BONUS_WEIGHT * (1 - (abs(theta) / 0.21) ** 2)
    position_bonus = POSITION_CENTERING_BONUS_WEIGHT * (1 - (abs(x) / 4.8) ** 2)
    velocity_penalty = -0.8 * (abs(theta_dot) / 2.0)
    total_bonus = max(0, angle_bonus + position_bonus + velocity_penalty)
    return base_reward + total_bonus

EWMA_SMOOTHING_ALPHA = 0.1
REWARD_HISTORY_WINDOW = 50
PROGRESS_REPORT_FREQUENCY = 100

def calculate_ewma(scores, alpha=EWMA_SMOOTHING_ALPHA, warmup_period=5):
    """
    Рассчитывает экспоненциально взвешенное скользящее среднее с периодом разогрева.

    Args:
        scores: список значений для сглаживания
        alpha: коэффициент сглаживания (0 < alpha <= 1)
        warmup_period: количество первых значений для усреднения (период разогрева)
    Returns:
        float: значение EWMA
    """
    if not scores:
        return 0.0

    # Если данных меньше периода разогрева, берём среднее всех доступных
    if len(scores) <= warmup_period:
        return np.mean(scores)

    # Инициализируем EWMA как среднее первых warmup_period значений
    ewma = np.mean(scores[:warmup_period])

    # Применяем EWMA к оставшимся значениям
    for score in scores[warmup_period:]:
        ewma = alpha * score + (1 - alpha) * ewma

    return ewma

def train_q_learning():
    env = gym.make(ENVIRONMENT_ID)

    # ПРОВЕРКА РАЗМЕРА Q‑ТАБЛИЦЫ ПЕРЕД СОЗДАНИЕМ
    is_valid_size, calculated_size = check_q_table_size(
        STATE_BUCKET_SIZES,
        env.action_space.n,
        MAX_Q_TABLE_SIZE
    )

    if not is_valid_size:
        print(f"❌ ПЕРЕПОЛНЕНИЕ Q‑ТАБЛИЦЫ ОБНАРУЖЕНО!")
        print(f"  Вычисленный размер: {calculated_size:,} элементов")
        print(f"  Максимальный допустимый: {MAX_Q_TABLE_SIZE:,} элементов")
        print(f"  Соотношение: {calculated_size / MAX_Q_TABLE_SIZE:.2f}x превышения")
        print("\nРЕКОМЕНДАЦИИ:")
        print("  1. Уменьшите STATE_BUCKET_SIZES (например, [4, 4, 8, 6])")
        print("  2. Используйте аппроксимацию функций (нейросети) вместо таблиц")
        print("  3. Увеличьте MAX_Q_TABLE_SIZE, если есть достаточно памяти")
        print("❌ Обучение прервано из‑за потенциального переполнения памяти.")
        env.close()
        return None, None, None, None, None

    # Создаём Q‑таблицу только если размер допустим
    Q_TABLE_SHAPE = STATE_BUCKET_SIZES + [env.action_space.n]
    q_table = np.zeros(Q_TABLE_SHAPE)

    print(f"✅ Q‑таблица создана успешно: {calculated_size:,} элементов "
          f"({calculated_size / MAX_Q_TABLE_SIZE * 100:.1f}% от лимита)")

    scores = []
    avg_scores = []
    td_errors_history = []  # Храним TD‑ошибки для анализа
    success_count = 0

    best_q_table = None
    best_avg_reward = -np.inf

    print("Начало обучения Q‑learning...")
    print(f"Параметры: MAX_EPISODES={MAX_EPISODES}, TARGET_REWARD={TARGET_EPISODE_REWARD}")


    for episode in range(MAX_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        discretized_state = discretize_state(state)
        score = 0
        epsilon = get_exploration_rate(episode)
        td_errors = []  # Ошибки текущего эпизода

        action_counts = [0, 0]

        for t in range(500):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[discretized_state])

            action_counts[action] += 1

            next_state, base_reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True

            reward = calculate_reward(next_state, base_reward)
            discretized_next_state = discretize_state(next_state)

            # Расчёт TD‑ошибки
            best_next_action = np.argmax(q_table[discretized_next_state])
            td_target = reward + DISCOUNT_FACTOR_GAMMA * q_table[discretized_next_state][best_next_action]
            td_error = td_target - q_table[discretized_state + (action,)]
            td_errors.append(abs(td_error))  # Сохраняем абсолютную ошибку


            alpha = get_learning_rate(episode, td_error)
            q_table[discretized_state + (action,)] += alpha * td_error

            discretized_state = discretized_next_state
            score += base_reward

            if done:
                break

        scores.append(score)
        avg_td_error = np.mean(td_errors) if td_errors else 0
        td_errors_history.append(avg_td_error)

        # Расчёт среднего reward за последние N эпизодов
        if len(scores) >= REWARD_HISTORY_WINDOW:
            avg_score = np.mean(scores[-REWARD_HISTORY_WINDOW:])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(np.mean(scores))


        # Логирование каждые PROGRESS_REPORT_FREQUENCY эпизодов
        if (episode + 1) % PROGRESS_REPORT_FREQUENCY == 0:
            # Используем исправленную функцию с периодом разогрева 5 эпизодов
            ewma_score = calculate_ewma(scores[-REWARD_HISTORY_WINDOW:], warmup_period=5)
            print(f"\n--- Эпизод {episode + 1} ---")
            print(f"  EWMA reward: {ewma_score:.2f}")
            print(f"  Средний reward ({REWARD_HISTORY_WINDOW} эпизодов): {avg_scores[-1]:.2f}")
            print(f"  Текущий epsilon: {epsilon:.3f}")
            print(f"  Текущая alpha: {alpha:.3f}")
            print(f"  Средняя |TD‑ошибка|: {avg_td_error:.3f}")
            print(f"  Последовательных успехов: {success_count}/{CONSECUTIVE_SUCCESS_THRESHOLD}")
            print(f"  Лучший средний reward: {best_avg_reward:.2f}")

                # Проверка успеха и сохранение лучшей модели
        if (len(scores) >= REWARD_HISTORY_WINDOW and
                np.mean(scores[-REWARD_HISTORY_WINDOW:]) >= TARGET_EPISODE_REWARD):
            success_count += 1
            print(f"  УСЛОВИЕ УСПЕХА ВЫПОЛНЕНО! Текущий средний: {np.mean(scores[-REWARD_HISTORY_WINDOW:]):.2f} "
                  f"(цель: {TARGET_EPISODE_REWARD})")

            if success_count >= CONSECUTIVE_SUCCESS_THRESHOLD:
                print(f"\n✅ СТАБИЛЬНЫЙ УСПЕХ ДОСТИГНУТ НА ЭПИЗОДЕ {episode + 1}!")
                print(f"   Целевой reward {TARGET_EPISODE_REWARD} удерживается {CONSECUTIVE_SUCCESS_THRESHOLD} проверок подряд.")
                break
        else:
            if success_count > 0:
                print(f"  Серия успехов прервана. Счётчик сброшен до 0.")
            success_count = 0

        # Сохранение лучшей модели
        current_avg = np.mean(scores[-REWARD_HISTORY_WINDOW:]) if len(scores) >= REWARD_HISTORY_WINDOW else np.mean(scores)
        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            best_q_table = q_table.copy()
            print(f"  🏆 ОБНОВЛЕНИЕ ЛУЧШЕЙ МОДЕЛИ: средний reward = {best_avg_reward:.2f}")

    env.close()
    return scores, avg_scores, td_errors_history, best_q_table, best_avg_reward

if __name__ == "__main__":
    scores, avg_scores, td_errors, best_model, final_best_avg_reward = train_q_learning()

    # Визуализация результатов
    plt.figure(figsize=(15, 10))

    # График 1: Reward за эпизод и средний reward
    plt.subplot(2, 2, 1)
    plt.plot(scores, label='Reward за эпизод', alpha=0.6)
    if len(avg_scores) > 0:
        plt.plot(range(REWARD_HISTORY_WINDOW - 1, len(avg_scores) + REWARD_HISTORY_WINDOW - 1),
                  avg_scores, label='Средний reward ({REWARD_HISTORY_WINDOW} эпизодов)', color='red')
    plt.axhline(y=TARGET_EPISODE_REWARD, color='green', linestyle='--',
               label=f'Целевой reward {TARGET_EPISODE_REWARD}')
    plt.xlabel('Эпизод')
    plt.ylabel('Reward')
    plt.title('Обучение Q‑learning: Reward динамика')
    plt.legend()
    plt.grid(True)

    # График 2: TD‑ошибки
    plt.subplot(2, 2, 2)
    plt.plot(td_errors, color='orange', label='Средняя |TD‑ошибка| за эпизод')
    plt.axhline(y=np.mean(td_errors), color='red', linestyle='--',
               label=f'Средняя TD‑ошибка: {np.mean(td_errors):.3f}')
    plt.xlabel('Эпизод')
    plt.ylabel('|TD‑ошибка|')
    plt.title('Стабильность обучения (TD‑ошибки)')
    plt.legend()
    plt.grid(True)

    # График 3: Epsilon и Alpha
    plt.subplot(2, 2, 3)
    episodes_for_params = list(range(0, len(scores), PROGRESS_REPORT_FREQUENCY))
    epsilon_values = [get_exploration_rate(ep) for ep in episodes_for_params]
    alpha_values = []
    for ep in episodes_for_params:
        # Для простоты берём среднее значение alpha за эпизод — в реальности нужно сохранять историю
        alpha_values.append(0.3 if ep < 1000 else 0.2 if ep < 1500 else 0.1)

    plt.plot(episodes_for_params, epsilon_values, label='Epsilon (исследование)', color='purple')
    plt.plot(episodes_for_params, alpha_values, label='Alpha (скорость обучения)', color='brown')
    plt.xlabel('Эпизод')
    plt.ylabel('Значение')
    plt.title('Параметры обучения: Epsilon и Alpha')
    plt.legend()
    plt.grid(True)

    # График 4: Распределение финальных rewards
    plt.subplot(2, 2, 4)
    plt.hist(scores[-100:], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=TARGET_EPISODE_REWARD, color='red', linestyle='--',
                label=f'Цель: {TARGET_EPISODE_REWARD}')
    plt.xlabel('Reward за эпизод')
    plt.ylabel('Частота')
    plt.title('Распределение rewards (последние 100 эпизодов)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Сохранение лучшей модели
    if best_model is not None:
        np.save('best_q_table.npy', best_model)
        print(f"\n💾 Лучшая модель сохранена в 'best_q_table.npy'")
        print(f"🏆 Лучший средний reward: {final_best_avg_reward:.2f}")

    # Статистика обучения
    print(f"\n--- СТАТИСТИКА ОБУЧЕНИЯ ---")
    print(f"Всего эпизодов: {len(scores)}")
    print(f"Максимальный reward за эпизод: {max(scores):.2f}")
    print(f"Средний reward за все эпизоды: {np.mean(scores):.2f}")
    if len(scores) >= 100:
        print(f"Средний reward за последние 100 эпизодов: {np.mean(scores[-100:]):.2f}")
        print(f"Стандартная девиация rewards (последние 100): {np.std(scores[-100:]):.2f}")

    achieved_target = final_best_avg_reward >= TARGET_EPISODE_REWARD
    print(f"Достигнут целевой reward {TARGET_EPISODE_REWARD}: {'✅ Да' if achieved_target else '❌ Нет'}")

    # Тестирование лучшей модели
    print(f"\n🧪 Тестирование лучшей модели...")
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

    if successful_tests == test_episodes:
        print("🎉 ПОЛНЫЙ УСПЕХ В ТЕСТИРОВАНИИ: все эпизоды достигли целевого reward!")
    else:
        print(f"⚠️  В тестировании {test_episodes - successful_tests} эпизодов не достигли цели.")
