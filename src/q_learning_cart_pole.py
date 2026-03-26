import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import lru_cache

# КОНФИГУРАЦИЯ — все параметры в одном месте
CONFIG = {
    # Основные параметры обучения
    'EPISODES': 1000,
    'RENDER_EVERY': 20,

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
    'SAVE_MODEL_EVERY': 100,

    # ПАРАМЕТРЫ ДЛЯ АНАЛИЗА ПРОГРЕССА
    'PROGRESS_WINDOW': 50,         # окно для анализа прогресса
    'PROGRESS_THRESHOLD': 0.2,    # порог для адаптации
}


class QLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.data_path = self._setup_directories()
        self.observation_high, self.observation_low, self.action_space_n = self._initialize_environment()
        self.q_table = self._create_q_table()
        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self.epsilons_history = []

        # Конвертируем в кортежи один раз для кэширования
        self.OBSERVATION_LOW_TUPLE = tuple(self.observation_low)
        self.OBSERVATION_HIGH_TUPLE = tuple(self.observation_high)
        self.DISCRETE_OS_SIZE_TUPLE = tuple(self.config['DISCRETE_OS_SIZE'])

    def _setup_directories(self):
        """Создаёт необходимые директории для сохранения данных."""
        current_path = Path(__file__).parent
        project_root = current_path.parent
        data_path = project_root / 'data'
        data_path.mkdir(exist_ok=True, parents=True)
        print(f"Папка создана по пути: {data_path}")
        return data_path

    def _initialize_environment(self):
        """Инициализация окружения и получение границ наблюдений."""
        env = gym.make('CartPole-v1')
        observation_high = np.array(env.observation_space.high, dtype=np.float64)
        observation_low = np.array(env.observation_space.low, dtype=np.float64)
        action_space_n = env.action_space.n
        env.close()

        # Заменяем бесконечные значения на фиксированные границы для нормализации
        for i in range(len(observation_high)):
            if np.isinf(observation_high[i]):
                observation_high[i] = 5.0
            if np.isinf(observation_low[i]):
                observation_low[i] = -5.0

        return observation_high, observation_low, action_space_n

    def _create_q_table(self):
        """Создание Q‑таблицы."""
        return np.zeros(
            tuple(self.config['DISCRETE_OS_SIZE']) + (self.action_space_n,)
        )

    @lru_cache(maxsize=1000)
    def get_discrete_state_cached(self, state_tuple):
        """Кэшированная версия преобразования непрерывного состояния в дискретное."""
        state = np.array(state_tuple, dtype=np.float64)

        if state.size != len(self.OBSERVATION_LOW_TUPLE):
            raise ValueError(
                f"Размер state ({state.size}) не соответствует ожидаемому ({len(self.OBSERVATION_LOW_TUPLE)})"
            )

        valid_range = np.isfinite(self.observation_high) & np.isfinite(self.observation_low)
        normalized = np.zeros_like(state)

        if np.any(valid_range):
            range_values = self.observation_high[valid_range] - self.observation_low[valid_range]
            range_values = np.where(range_values == 0, 1.0, range_values)
            normalized[valid_range] = (
                (state[valid_range] - self.observation_low[valid_range]) / range_values
            )
            normalized[valid_range] = np.clip(normalized[valid_range], 0.0, 1.0)

        infinite_mask = ~valid_range
        if np.any(infinite_mask):
            normalized[infinite_mask] = 1 / (1 + np.exp(-state[infinite_mask]))

        discrete_state = (normalized * self.config['DISCRETE_OS_SIZE']).astype(np.int_)
        discrete_state = np.clip(discrete_state, 0, np.array(self.config['DISCRETE_OS_SIZE']) - 1)

        return tuple(discrete_state)

    def calculate_progress(self, rewards, window):
        """Рассчитывает прогресс как разницу средних наград между последними и предыдущими эпизодами."""
        if len(rewards) < window * 2:
            return 0
        recent = rewards[-window:]  # последние эпизоды
        previous = rewards[-window*2:-window]  # эпизоды перед последними
        return np.mean(recent) - np.mean(previous)

    def print_cache_stats(self):
        """Выводит статистику по кэшу."""
        cache_info = self.get_discrete_state_cached.cache_info()
        print(f"Кэш дискретных состояний: hits={cache_info.hits}, "
              f"misses={cache_info.misses}, "
              f"current_size={cache_info.currsize}/{cache_info.maxsize}")

    def train_episode(self, episode):
        """Обучает один эпизод и возвращает накопленную награду."""
        render_mode = 'human' if episode % self.config['RENDER_EVERY'] == 0 else None
        env = gym.make('CartPole-v1', render_mode=render_mode)

        episode_reward = 0
        state, info = env.reset()
        discrete_state = self.get_discrete_state_cached(tuple(state))

        done = False

        while not done:
            # Выбор действия: ε‑жадная стратегия
            if np.random.random() > self.config['EPSILON']:
                action = np.argmax(self.q_table[discrete_state])
            else:
                action = np.random.randint(0, self.action_space_n)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            new_discrete_state = self.get_discrete_state_cached(tuple(new_state))

            # Обновление Q‑таблицы только если не финальное состояние
            if not done:
                max_future_q = np.max(self.q_table[new_discrete_state])
                current_q = self.q_table[discrete_state + (action,)]
                new_q = (1 - self.config['LEARNING_RATE']) * current_q + self.config['LEARNING_RATE'] * (reward + self.config['DISCOUNT'] * max_future_q)
                self.q_table[discrete_state + (action,)] = new_q
            else:
                # В финальном состоянии Q‑значение = 0
                self.q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        env.close()  # Закрываем среду после каждого эпизода
        return episode_reward

    def update_epsilon(self, episode):
        """Обновляет значение эпсилона согласно стратегии затухания."""
        if episode >= self.config['START_EPSILON_DECAYING']:
            decay_steps = episode - self.config['START_EPSILON_DECAYING']
            self.config['EPSILON'] = (
                self.config['MIN_EPSILON'] +
                (1.0 - self.config['MIN_EPSILON']) *
                np.exp(-self.config['EPSILON_DECAY_RATE'] * decay_steps)
            )
            # Гарантируем, что эпсилон не опустится ниже минимума
            self.config['EPSILON'] = max(self.config['EPSILON'], self.config['MIN_EPSILON'])

    def save_model(self, episode):
        """Сохраняет текущую Q‑таблицу в файл."""
        model_path = self.data_path / f"q_table_episode_{episode}.npy"
        np.save(model_path, self.q_table)
        print(f"Модель сохранена: {model_path}")

    def log_episode_stats(self, episode, avg_reward, min_reward, max_reward):
        """Логирует статистику эпизода."""
        print(f"Эпизод {episode}: avg reward: {avg_reward:.2f}, "
              f"min: {min_reward:.2f}, max: {max_reward:.2f}, "
              f"epsilon: {self.config['EPSILON']:.3f}")

    def should_stop_training(self, progress):
        """Проверяет, нужно ли остановить обучение."""
        if len(self.ep_rewards) >= self.config['PROGRESS_WINDOW']:
            # Если цель достигнута и прогресс отсутствует — останавливаем обучение
            if (progress < self.config['PROGRESS_THRESHOLD'] and
                self.aggr_ep_rewards['avg'][-1] >= self.config['TARGET_REWARD']):
                print(f"Цель достигнута на эпизоде {len(self.ep_rewards)}! Обучение завершено.")
                return True

            # Дополнительная проверка: если эпсилон достиг минимума и прогресс отсутствует
            if (self.config['EPSILON'] <= self.config['MIN_EPSILON'] + 1e-6 and
                progress < self.config['PROGRESS_THRESHOLD'] / 2 and
                len(self.ep_rewards) > self.config['START_EPSILON_DECAYING'] + self.config['PROGRESS_WINDOW']):
                print(f"Обучение стабилизировалось на эпизоде {len(self.ep_rewards)}. Остановка.")
                return True
        return False

    def plot_training_results(self):
        """Строит и сохраняет графики результатов обучения."""
        plt.figure(figsize=(12, 8))

        # График наград
        plt.subplot(2, 1, 1)
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['avg'], label='Среднее', color='blue')
        plt.fill_between(
            self.aggr_ep_rewards['ep'],
            self.aggr_ep_rewards['min'],
            self.aggr_ep_rewards['max'],
            alpha=0.3,
            label='Диапазон',
            color='blue'
        )
        plt.axhline(y=self.config['TARGET_REWARD'], color='red', linestyle='--', label='Цель')
        plt.title('Обучение: награды по эпизодам')
        plt.ylabel('Награда')
        plt.legend()

        # График эпсилона
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.epsilons_history)), self.epsilons_history, color='green')
        plt.title('Эпсилон по эпизодам (экспоненциальное затухание)')
        plt.xlabel('Эпизод')
        plt.ylabel('Эпсилон')

        plt.tight_layout()
        plt.savefig(self.data_path / 'training_results.png')
        print(f"Графики сохранены: {self.data_path / 'training_results.png'}")
        plt.show()

    def train(self):
        """Основной цикл обучения."""
        for episode in range(self.config['EPISODES']):
            # Обучение одного эпизода
            episode_reward = self.train_episode(episode)
            self.ep_rewards.append(episode_reward)
            self.epsilons_history.append(self.config['EPSILON'])

            # Обновление эпсилона
            self.update_epsilon(episode)

            # Сбор статистики каждые RENDER_EVERY эпизодов
            if episode % self.config['RENDER_EVERY'] == 0 and episode > 0:
                recent_rewards = self.ep_rewards[-self.config['RENDER_EVERY']:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                min_reward = min(recent_rewards)
                max_reward = max(recent_rewards)

                self.aggr_ep_rewards['ep'].append(episode)
                self.aggr_ep_rewards['avg'].append(avg_reward)
                self.aggr_ep_rewards['min'].append(min_reward)
                self.aggr_ep_rewards['max'].append(max_reward)

                self.log_episode_stats(episode, avg_reward, min_reward, max_reward)

            # Сохранение модели каждые SAVE_MODEL_EVERY эпизодов
            if episode % self.config['SAVE_MODEL_EVERY'] == 0:
                self.save_model(episode)

            # Проверка условий остановки
            progress = self.calculate_progress(self.ep_rewards, self.config['PROGRESS_WINDOW'])
            if self.should_stop_training(progress):
                break

            # Вывод статистики кэша каждые 100 эпизодов
            if episode % 100 == 0:
                self.print_cache_stats()

        # Построение графиков после завершения обучения
        self.plot_training_results()


# Запуск обучения
if __name__ == '__main__':
    trainer = QLearningTrainer(CONFIG)
    trainer.train()
    