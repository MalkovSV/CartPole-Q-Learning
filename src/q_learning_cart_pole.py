import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# КОНФИГУРАЦИЯ — все параметры в одном месте
CONFIG = {
    # Основные параметры обучения
    'EPISODES': 1000,
    'RENDER_EVERY': 10,

    # Параметры Q‑learning
    'LEARNING_RATE': 0.1,
    'DISCOUNT': 0.95,

    # ε‑жадная стратегия
    'EPSILON': 1.0,
    'START_EPSILON_DECAYING': 100,      # Начинаем затухание после 100 эпизодов
    'MIN_EPSILON': 0.1,             # Минимальное значение эпсилона
    'EPSILON_DECAY_RATE': 0.01,   # Коэффициент экспоненциального затухания (λ)

    # Дискретизация пространства состояний
    'DISCRETE_OS_SIZE': [20] * 4,  # CartPole имеет 4 измерения состояния

    # Целевые показатели
    'TARGET_REWARD': 450,

    # Сохранение моделей
    'SAVE_MODEL_EVERY': 50,

    # ПАРАМЕТРЫ ДЛЯ АНАЛИЗА ПРОГРЕССА
    'PROGRESS_WINDOW': 50,         # окно для анализа прогресса (не используется в экспоненциальном затухании напрямую)
    'PROGRESS_THRESHOLD': 0.2,    # порог для адаптации (не используется в базовой версии экспоненциального затухания)
}

# Путь к текущей директории
current_path = Path(__file__).parent
# Путь к корню проекта
project_root = current_path.parent
# Полный путь к папке data
data_path = project_root / 'data'

# Создаём папку (parents=True позволяет создавать промежуточные папки)
data_path.mkdir(exist_ok=True, parents=True)
print(f"Папка создана по пути: {data_path}")

def get_discrete_state(state, observation_low, observation_high, discrete_os_size):
    """
    Преобразует непрерывное состояние в дискретное с улучшенной обработкой крайних случаев.

    Args:
        state: непрерывное состояние (может быть tuple или array)
        observation_low: нижние границы пространства наблюдений
        observation_high: верхние границы пространства наблюдений
        discrete_os_size: размер дискретного пространства для каждого измерения

    Returns:
        tuple: дискретное состояние
    """
    # Обработка tuple (если возвращается из reset)
    if isinstance(state, tuple):
        state = state[0]

    state_array = np.array(state, dtype=np.float64)

    # Проверка размеров
    if state_array.size != len(observation_low):
        raise ValueError(
            f"Размер state ({state_array.size}) не соответствует ожидаемому ({len(observation_low)})"
        )

    # Нормализация с учётом границ
    # Обрабатываем бесконечные границы
    valid_range = np.isfinite(observation_high) & np.isfinite(observation_low)
    normalized = np.zeros_like(state_array)

    # Для измерений с конечными границами — нормальная нормализация
    if np.any(valid_range):
        range_values = observation_high[valid_range] - observation_low[valid_range]
        # Избегаем деления на ноль
        range_values = np.where(range_values == 0, 1.0, range_values)
        normalized[valid_range] = (
            (state_array[valid_range] - observation_low[valid_range]) / range_values
        )
        # Ограничиваем нормализованные значения диапазоном [0, 1]
        normalized[valid_range] = np.clip(normalized[valid_range], 0.0, 1.0)

    # Для измерений с бесконечными границами — используем сигмоиду для сжатия
    infinite_mask = ~valid_range
    if np.any(infinite_mask):
        # Сигмоида сжимает любые значения в диапазон (0,1)
        normalized[infinite_mask] = 1 / (1 + np.exp(-state_array[infinite_mask]))

    # Преобразование в дискретные индексы
    discrete_state = (normalized * discrete_os_size).astype(np.int_)
    # Гарантируем, что индексы в допустимом диапазоне
    discrete_state = np.clip(discrete_state, 0, np.array(discrete_os_size) - 1)

    return tuple(discrete_state)

# ФУНКЦИЯ РАСЧЁТА ПРОГРЕССА ОБУЧЕНИЯ
def calculate_progress(rewards, window):
    """
    Рассчитывает прогресс как разницу средних наград между последними и предыдущими эпизодами.

    Args:
        rewards: список наград за все эпизоды
        window: размер окна для анализа (количество эпизодов)

    Returns:
        progress: разница средних значений (может быть отрицательной)
    """
    if len(rewards) < window * 2:
        return 0
    recent = rewards[-window:]  # последние эпизоды
    previous = rewards[-window*2:-window]  # эпизоды перед последними
    return np.mean(recent) - np.mean(previous)

# Переменные для хранения данных среды
observation_high = None
observation_low = None
discrete_os_win_size = None
q_table = None
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
epsilons_history = []  # для графика эпсилона

# ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
for episode in range(CONFIG['EPISODES']):
    # Настройка рендеринга
    render_mode = 'human' if episode % CONFIG['RENDER_EVERY'] == 0 else None
    env = gym.make('CartPole-v1', render_mode=render_mode)

    if episode == 0:
        observation_high = np.array(env.observation_space.high, dtype=np.float64)
        observation_low = np.array(env.observation_space.low, dtype=np.float64)

    # Сохраняем оригинальные границы для дискретизации
    original_observation_high = observation_high.copy()
    original_observation_low = observation_low.copy()

    # Заменяем бесконечные значения на фиксированные границы для нормализации
    for i in range(len(observation_high)):
        if np.isinf(observation_high[i]):
            observation_high[i] = 5.0
        if np.isinf(observation_low[i]):
            observation_low[i] = -5.0

    q_table = np.zeros(
        tuple(CONFIG['DISCRETE_OS_SIZE']) + (env.action_space.n,)
    )

    episode_reward = 0
    state, info = env.reset()
    discrete_state = get_discrete_state(
        state,
        original_observation_low,  # Используем оригинальные границы
        original_observation_high,
        CONFIG['DISCRETE_OS_SIZE']
)

    done = False

    while not done:
        # Выбор действия: ε‑жадная стратегия
        if np.random.random() > CONFIG['EPSILON']:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        new_discrete_state = get_discrete_state(
            new_state,
            original_observation_low,
            original_observation_high,
            CONFIG['DISCRETE_OS_SIZE']
        )


        # Обновление Q‑таблицы только если не финальное состояние
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - CONFIG['LEARNING_RATE']) * current_q + CONFIG['LEARNING_RATE'] * (reward + CONFIG['DISCOUNT'] * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        else:
            # В финальном состоянии Q‑значение = 0
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

        ep_rewards.append(episode_reward)
    epsilons_history.append(CONFIG['EPSILON'])

    # ЭКПОНЕНЦИАЛЬНОЕ ЗАТУХАНИЕ ЭПСИЛОНА
    if episode >= CONFIG['START_EPSILON_DECAYING']:
        decay_steps = episode - CONFIG['START_EPSILON_DECAYING']
        CONFIG['EPSILON'] = (
            CONFIG['MIN_EPSILON'] +
            (1.0 - CONFIG['MIN_EPSILON']) *
            np.exp(-CONFIG['EPSILON_DECAY_RATE'] * decay_steps)
        )
        # Гарантируем, что эпсилон не опустится ниже минимума
        CONFIG['EPSILON'] = max(CONFIG['EPSILON'], CONFIG['MIN_EPSILON'])

    # Сбор статистики каждые RENDER_EVERY эпизодов
    if episode % CONFIG['RENDER_EVERY'] == 0 and episode > 0:
        # Расчёт агрегированных метрик за последние RENDER_EVERY эпизодов
        recent_rewards = ep_rewards[-CONFIG['RENDER_EVERY']:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        min_reward = min(recent_rewards)
        max_reward = max(recent_rewards)

        # Сохранение агрегированных данных
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min_reward)
        aggr_ep_rewards['max'].append(max_reward)

        print(f"Эпизод {episode}: avg reward: {avg_reward:.2f}, "
              f"min: {min_reward:.2f}, max: {max_reward:.2f}, "
              f"epsilon: {CONFIG['EPSILON']:.3f}")

    # Сохранение модели каждые SAVE_MODEL_EVERY эпизодов
    if episode % CONFIG['SAVE_MODEL_EVERY'] == 0:
        model_path = data_path / f"q_table_episode_{episode}.npy"
        np.save(model_path, q_table)
        print(f"Модель сохранена: {model_path}")

# Завершение работы среды
env.close()

# Построение графиков после завершения обучения
plt.figure(figsize=(12, 8))

# График наград
plt.subplot(2, 1, 1)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='Среднее', color='blue')
plt.fill_between(
    aggr_ep_rewards['ep'],
    aggr_ep_rewards['min'],
    aggr_ep_rewards['max'],
    alpha=0.3,
    label='Диапазон',
    color='blue'
)
plt.axhline(y=CONFIG['TARGET_REWARD'], color='red', linestyle='--', label='Цель')
plt.title('Обучение: награды по эпизодам')
plt.ylabel('Награда')
plt.legend()

# График эпсилона
plt.subplot(2, 1, 2)
plt.plot(range(len(epsilons_history)), epsilons_history, color='green')
plt.title('Эпсилон по эпизодам (экспоненциальное затухание)')
plt.xlabel('Эпизод')
plt.ylabel('Эпсилон')

plt.tight_layout()
plt.show()

# Сохранение графиков
plt.savefig(data_path / 'training_results.png')
print(f"Графики сохранены: {data_path / 'training_results.png'}")
