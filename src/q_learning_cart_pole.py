import gymnasium as gym  # вместо gym, актуальная версия
import numpy as np
import matplotlib.pyplot as plt

# Параметры
TARGET_REWARD = 450
EPISODES = 3000
DISCRETE_OS_SIZE = [8, 8, 12, 10]
STATE_BOUNDS = list(zip(
    [-4.8, -3.5, -0.21, -2],
    [4.8, 3.5, 0.21, 2]
))

# Гиперпараметры Q‑learning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class QLearningAgent:
    def __init__(self, discrete_os_size, state_bounds):
        self.discrete_os_size = discrete_os_size
        self.state_bounds = state_bounds
        # Инициализация Q‑таблицы нулями
        self.q_table = np.zeros(self.discrete_os_size + [2])  # 2 действия: 0 — влево, 1 — вправо
        self.epsilon = EPSILON

    def get_discrete_state(self, state):
        """Преобразуем непрерывное состояние в дискретное"""
        discrete_state = []
        for i in range(len(state)):
            # Нормализуем значение в диапазоне [0, DISCRETE_OS_SIZE[i]-1]
            normalized = (state[i] - self.state_bounds[i][0]) / \
                        (self.state_bounds[i][1] - self.state_bounds[i][0])
            discrete = int(normalized * (self.discrete_os_size[i] - 1))
            # Ограничиваем значения допустимыми индексами
            discrete = max(0, min(discrete, self.discrete_os_size[i] - 1))
            discrete_state.append(discrete)
        return tuple(discrete_state)

    def choose_action(self, discrete_state):
        """Выбор действия: ε‑жадная стратегия"""
        if np.random.random() > self.epsilon:
            # Эксплуатация: выбираем действие с максимальным Q‑значением
            action = np.argmax(self.q_table[discrete_state])
        else:
            # Исследование: случайное действие
            action = np.random.randint(0, 2)
        return action

    def update_q_value(self, discrete_state, action, reward, new_discrete_state, done):
        """Обновление Q‑значения по формуле Q‑learning"""
        current_q = self.q_table[discrete_state + (action,)]

        if not done:
            max_future_q = np.max(self.q_table[new_discrete_state])
            new_q = current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q - current_q
            )
        else:
            new_q = reward

        self.q_table[discrete_state + (action,)] = new_q

    def decay_epsilon(self):
        """Уменьшение ε для баланса exploration‑exploitation"""
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

def train_agent():
    # Создаём среду
    env = gym.make('CartPole-v1')
    agent = QLearningAgent(DISCRETE_OS_SIZE, STATE_BOUNDS)

    # Для отслеживания прогресса
    episode_rewards = []
    best_reward = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        discrete_state = agent.get_discrete_state(state)
        episode_reward = 0
        done = False

        while not done:
            # Выбор действия
            action = agent.choose_action(discrete_state)
            # Выполнение действия
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Дискретизация нового состояния
            new_discrete_state = agent.get_discrete_state(new_state)

            # Обновление Q‑таблицы
            agent.update_q_value(discrete_state, action, reward, new_discrete_state, done)

            # Переход к следующему состоянию
            discrete_state = new_discrete_state

        # Уменьшение ε
        agent.decay_epsilon()

        # Сохранение награды эпизода
        episode_rewards.append(episode_reward)
        best_reward = max(best_reward, episode_reward)

        # Вывод прогресса каждые 100 эпизодов
        if episode % 100 == 0:
            print(f"Эпизод {episode}, награда: {episode_reward}, ε: {agent.epsilon:.3f}")

        # Проверка достижения цели
        if best_reward >= TARGET_REWARD:
            print(f"\nЦель достигнута на эпизоде {episode}! Лучшая награда: {best_reward}")
            break

    env.close()
    return episode_rewards, agent

def plot_results(episode_rewards):
    """Визуализация результатов обучения"""
    plt.figure(figsize=(12, 6))

    # График наград по эпизодам
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Награды по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')

    # Скользящее среднее
    plt.subplot(1, 2, 2)
    moving_avg = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg)
    plt.title('Скользящее среднее (50 эпизодов)')
    plt.xlabel('Эпизод')
    plt.ylabel('Средняя награда')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Обучение агента
    rewards, trained_agent = train_agent()
    # Визуализация результатов
    plot_results(rewards)
