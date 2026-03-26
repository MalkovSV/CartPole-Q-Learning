import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Путь к текущей директории
current_path = Path(__file__).parent
# Путь к корню проекта
project_root = current_path.parent
# Полный путь к папке data
data_path = project_root / 'data'

# Создаём папку (parents=True позволяет создавать промежуточные папки)
data_path.mkdir(exist_ok=True, parents=True)
print(f"Папка создана по пути: {data_path}")

def get_discrete_state(state, observation_low, discrete_os_win_size, DISCRETE_OS_SIZE):
    """Преобразует непрерывное состояние в дискретное"""
    # Если state — кортеж, берём первый элемент (массив состояния)
    if isinstance(state, tuple):
        state = state[0]

    try:
        state_array = np.array(state, dtype=np.float64)
        if state_array.size != len(observation_low):
            raise ValueError(
                f"Размер state ({state_array.size}) не соответствует ожидаемому ({len(observation_low)})"
            )
    except Exception as e:
        print(f"Ошибка преобразования state: {e}")
        print(f"Исходный state: {state}")
        state_array = np.zeros(len(observation_low), dtype=np.float64)

    # Безопасное вычисление дискретного состояния
    discrete_state = np.floor(
        (state_array - observation_low) / discrete_os_win_size
    ).astype(np.int_)

    # Ограничение значений в допустимых пределах
    discrete_state = np.clip(
        discrete_state,
        0,
        np.array(DISCRETE_OS_SIZE) - 1
    )
    return tuple(discrete_state)

# Инициализация параметров
RENDER_EVERY = 10
EPISODES = 200

# Параметры обучения
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# ε-жадная стратегия
EPSILON = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

# Дискретизация пространства состояний
DISCRETE_OS_SIZE = [20] * 4  # CartPole имеет 4 измерения состояния

# Переменные для хранения данных среды (инициализируются один раз)
observation_high = None
observation_low = None
discrete_os_win_size = None
q_table = None
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
epsilon_decay_value = 0

# Основной цикл обучения с гибкой настройкой рендеринга
for episode in range(EPISODES):
    # Настройка рендеринга: только каждый RENDER_EVERY эпизод
    render_mode = 'human' if episode % RENDER_EVERY == 0 else None
    env = gym.make('CartPole-v1', render_mode=render_mode)

    # Инициализация среды и параметров (выполняется один раз)
    if episode == 0:
        observation_high = np.array(env.observation_space.high, dtype=np.float64)
        observation_low = np.array(env.observation_space.low, dtype=np.float64)

        # Безопасная обработка бесконечных границ
        for i in range(len(observation_high)):
            if np.isinf(observation_high[i]):
                observation_high[i] = 5.0
            if np.isinf(observation_low[i]):
                observation_low[i] = -5.0

        # Преобразуем DISCRETE_OS_SIZE в массив для согласованности типов (используем int)
        DISCRETE_OS_SIZE_ARRAY = np.array(DISCRETE_OS_SIZE, dtype=np.int_)

        # Безопасное вычисление размеров окон дискретизации
        discrete_os_win_size = np.divide(
            np.subtract(observation_high, observation_low, dtype=np.float64),
            DISCRETE_OS_SIZE_ARRAY,
            dtype=np.float64
        )

        # Обработка нулевых диапазонов (избегаем деления на ноль)
        discrete_os_win_size = np.where(
            discrete_os_win_size == 0,
            1.0,  # если диапазон нулевой, размер окна = 1
            discrete_os_win_size
        )

        # Инициализация Q‑таблицы (инициализируем нулями)
        q_table = np.zeros(
            tuple(DISCRETE_OS_SIZE) + (env.action_space.n,)
        )

        # Вычисление скорости убывания эпсилона с проверкой
        if END_EPSILON_DECAYING > START_EPSILON_DECAYING:
            epsilon_decay_value = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
        else:
            epsilon_decay_value = 0

    episode_reward = 0

    # reset() теперь возвращает (observation, info)
    state, info = env.reset()
    discrete_state = get_discrete_state(state, observation_low, discrete_os_win_size, DISCRETE_OS_SIZE)
    done = False

    while not done:
        # Выбор действия: ε‑жадная стратегия
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # step() теперь возвращает 5 значений
        new_state, reward, terminated, truncated, info = env.step(action)

        # done теперь зависит от terminated ИЛИ truncated
        done = terminated or truncated
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state, observation_low, discrete_os_win_size, DISCRETE_OS_SIZE)

        # Обновление Q‑таблицы только если не финальное состояние
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        else:
            # В финальном состоянии Q‑значение = 0
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Завершение эпизода
    ep_rewards.append(episode_reward)

    # Уменьшение эпсилона
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= epsilon_decay_value

    # Сбор статистики каждые RENDER_EVERY эпизодов
    if episode % RENDER_EVERY == 0 and episode > 0:
        recent_rewards = ep_rewards[-RENDER_EVERY:]
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(np.mean(recent_rewards))
        aggr_ep_rewards['min'].append(np.min(recent_rewards))
        aggr_ep_rewards['max'].append(np.max(recent_rewards))

        print(f"Эпизод: {episode}, эпсилон: {EPSILON:.3f}")
        print(f"Среднее за {len(recent_rewards)} эпизодов: {np.mean(recent_rewards):.2f}")

    env.close()

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='Среднее', linewidth=2)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='Минимум', alpha=0.7)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='Максимум', alpha=0.7)
plt.xlabel('Эпизод')
plt.ylabel('Награда')
plt.title('Обучение Q‑Learning: CartPole-v1')
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем горизонтальную линию — целевое значение награды для CartPole
plt.axhline(y=450, color='r', linestyle='--', alpha=0.7, label='Целевая награда (450)')

# Улучшаем отображение осей
plt.xlim(left=0)
plt.ylim(bottom=0)

# Добавляем область заливки вокруг среднего значения
plt.fill_between(
    aggr_ep_rewards['ep'],
    aggr_ep_rewards['min'],
    aggr_ep_rewards['max'],
    color='skyblue',
    alpha=0.2,
    label='Диапазон наград'
)

# Добавляем аннотацию с итоговыми результатами (исправленная версия)
if aggr_ep_rewards['avg']:
    final_avg = aggr_ep_rewards['avg'][-1]
    plt.annotate(
        f'Финальное среднее: {final_avg:.1f}',
        xy=(aggr_ep_rewards['ep'][-1], final_avg),
        xytext=(20, -30),
        textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        fontsize=10
    )

plt.tight_layout()
plt.show()

# Сохранение графика в файл
plt.savefig(os.path.join(data_path, 'q_learning_training_progress.png'), dpi=300, bbox_inches='tight')
print("График сохранён как 'data/q_learning_training_progress.png'")

# Дополнительная статистика
print("\n=== ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ ===")
print(f"Всего эпизодов: {EPISODES}")
print(f"Финальное значение эпсилона: {EPSILON:.4f}")

if ep_rewards:
    print(f"Максимальная награда за эпизод: {max(ep_rewards)}")
    print(f"Минимальная награда за эпизод: {min(ep_rewards)}")
    print(f"Средняя награда за все эпизоды: {np.mean(ep_rewards):.2f}")
    print(f"Стандартное отклонение награды: {np.std(ep_rewards):.2f}")

    # Анализ прогресса: сравнение начала и конца обучения
    initial_period = ep_rewards[:RENDER_EVERY] if len(ep_rewards) >= RENDER_EVERY else ep_rewards
    final_period = ep_rewards[-RENDER_EVERY:]
    print(f"\nПрогресс обучения:")
    print(f"  Начало: среднее {np.mean(initial_period):.2f} ± {np.std(initial_period):.2f}")
    print(f"  Конец:  среднее {np.mean(final_period):.2f} ± {np.std(final_period):.2f}")

# Сохранение модели
np.save(os.path.join(data_path, 'q_table.npy'), q_table)
print("Q‑таблица сохранена как 'data/q_table.npy'")
