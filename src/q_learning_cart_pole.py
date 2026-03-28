import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from functools import lru_cache

# КОНФИГУРАЦИЯ — все параметры в одном месте
CONFIG = {
    # Основные параметры обучения
    'EPISODES': 1000,
    'RENDER_EVERY': 40,

    # Параметры Q‑learning
    'LEARNING_RATE': 0.1,
    'DISCOUNT': 0.95,

    # ε‑жадная стратегия
    'EPSILON': 1.0,
    'START_EPSILON_DECAYING': 100,      # Начинаем затухание после 100 эпизодов
    'MIN_EPSILON': 0.05,             # Минимальное значение эпсилона
    'EPSILON_DECAY_RATE': 0.01,   # Коэффициент экспоненциального затухания (λ)

    # Дискретизация пространства состояний
    'DISCRETE_OS_SIZE': [15] * 4,  # CartPole имеет 4 измерения состояния
    'DISCRETIZATION_METHOD': 'sigmoid',  # 'linear' или 'sigmoid'


    # Целевые показатели
    'TARGET_REWARD': 450,

    # Сохранение моделей
    'SAVE_MODEL_EVERY': 40,

    # ПАРАМЕТРЫ ДЛЯ АНАЛИЗА ПРОГРЕССА
    'PROGRESS_WINDOW': 50,         # окно для анализа прогресса
    'PROGRESS_THRESHOLD': 0.1,    # порог для адаптации

    # ОГРАНИЧЕНИЕ НА РАЗМЕР Q‑ТАБЛИЦЫ
    'MAX_Q_TABLE_SIZE': 5000,      # Максимальный размер таблицы
    'PRUNE_THRESHOLD': 4000,     # Порог для запуска очистки (80 % от максимума)
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

        # Оставляем только отслеживание лучшего среднего результата
        self.best_avg_reward = float('-inf')  # Лучшее среднее значение награды
        self.best_avg_model_path = None      # Путь к лучшей модели по среднему

        # Проверяем корректность метода дискретизации
        if self.config['DISCRETIZATION_METHOD'] not in ['linear', 'sigmoid']:
            raise ValueError("DISCRETIZATION_METHOD должен быть 'linear' или 'sigmoid'")

        # Конвертируем в кортежи
        self.OBSERVATION_LOW_TUPLE = tuple(self.observation_low)
        self.OBSERVATION_HIGH_TUPLE = tuple(self.observation_high)
        self.DISCRETE_OS_SIZE_TUPLE = tuple(self.config['DISCRETE_OS_SIZE'])

        # Инициализируем кэш с учётом метода дискретизации
        if self.config['DISCRETIZATION_METHOD'] == 'linear':
            self.get_discrete_state = self._get_discrete_state_linear
        else:
            self.get_discrete_state = self.get_discrete_state_sigmoid

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
        """Создаёт Q‑таблицу как словарь вместо массива."""
        return {}  # Пустой словарь вместо np.zeros

    def _get_q_value(self, discrete_state, action):
        """Получает Q‑значение для состояния и действия. Создаёт запись, если её нет."""
        key = discrete_state + (action,)
        if key not in self.q_table:
            self.q_table[key] = 0.0
        return self.q_table[key]

    def _set_q_value(self, discrete_state, action, value):
        """Устанавливает Q‑значение для состояния и действия."""
        self.q_table[discrete_state + (action,)] = value    

    def get_discrete_state_sigmoid(self, state_tuple):
        """
        Преобразование непрерывного состояния в дискретное (sigmoid‑метод).
        Нормализует входные значения и применяет сигмоидальное преобразование
        для бесконечных границ.
        """
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

        discrete_state = (normalized * np.array(self.config['DISCRETE_OS_SIZE'])).astype(np.int_)

        discrete_state = np.clip(discrete_state, 0, np.array(self.config['DISCRETE_OS_SIZE']) - 1)

        return tuple(discrete_state)


    def calculate_progress(self, rewards, window):
        if len(rewards) < window:
            return [0.0] * len(rewards)

        progress_values = [0.0] * (window - 1)

        for i in range(window - 1, len(rewards)):
            half = window // 2
            start_first = max(0, i - window + 1)
            end_first = start_first + half
            start_second = end_first
            end_second = i + 1

            first_half = rewards[start_first:end_first]
            second_half = rewards[start_second:end_second]

            avg_first = np.mean(first_half) if first_half else 0.0
            avg_second = np.mean(second_half) if second_half else 0.0

            progress_values.append(avg_second - avg_first)

        return progress_values




    def train_episode(self, episode):
        """Обучает один эпизод и возвращает накопленную награду."""
        render_mode = 'human' if episode % self.config['RENDER_EVERY'] == 0 else None
        env = gym.make('CartPole-v1', render_mode=render_mode)

        episode_reward = 0
        state, info = env.reset()
        discrete_state = self.get_discrete_state(tuple(state))  # Универсальный вызов

        done = False

        while not done:
            # Выбор действия: ε‑жадная стратегия
            if np.random.random() > self.config['EPSILON']:
                # Получаем все Q‑значения для текущего состояния
                q_values = [
                    self._get_q_value(discrete_state, a)
                    for a in range(self.action_space_n)
                ]
                action = np.argmax(q_values)
            else:
                action = np.random.randint(0, self.action_space_n)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            new_discrete_state = self.get_discrete_state(tuple(new_state))

            # Обновление Q‑таблицы
            if not done:
                # Получаем максимальное будущее Q‑значение
                future_q_values = [
                    self._get_q_value(new_discrete_state, a)
                    for a in range(self.action_space_n)
                ]
                max_future_q = np.max(future_q_values)

                # Текущее Q‑значение
                current_q = self._get_q_value(discrete_state, action)

                # Новое Q‑значение по формуле Q‑learning
                new_q = current_q + self.config['LEARNING_RATE'] * (
                reward + self.config['DISCOUNT'] * max_future_q - current_q
                )

                # Сохраняем обновлённое значение
                self._set_q_value(discrete_state, action, new_q)
            else:
                # В финальном состоянии Q‑значение = 0
                self._set_q_value(discrete_state, action, 0)

            discrete_state = new_discrete_state

        env.close()
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

    def load_model(self, model_path):
        """Загружает Q‑таблицу из файла."""
        with open(model_path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Модель загружена: {model_path}")

    def load_best_average_model(self):
        """Загружает лучшую модель по среднему результату."""
        avg_files = list(self.data_path.glob("best_avg_q_table*.pkl"))

        if not avg_files:
            print("Нет модели по среднему для загрузки.")
            return False

        def extract_avg_from_filename(filepath):
            import re
            match = re.search(r'_avg_([0-9]+\.[0-9]{2})\.pkl', filepath.name)
            if match:
                return float(match.group(1))
            else:
                return float('-inf')

        sorted_files = sorted(avg_files, key=extract_avg_from_filename, reverse=True)
        try:
            best_avg_path = sorted_files[0]
            avg_reward = extract_avg_from_filename(best_avg_path)

            # Загружаем модель
            self.load_model(best_avg_path)
            self.best_avg_reward = avg_reward
            self.best_avg_model_path = best_avg_path

            print(f"Загружена лучшая модель по среднему: {best_avg_path.name} (среднее: {avg_reward:.2f})")
            return True
        except Exception as e:
            print(f"Ошибка загрузки лучшей средней модели: {e}")
            return False

             

    def save_top_models(self, episode, avg_reward):
        """Сохраняет текущую модель, если среднее значение награды лучшее.

        Args:
            episode: номер эпизода
            avg_reward: среднее значение награды за окно эпизодов
        """
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            avg_path = self.data_path / f"best_avg_q_table_episode_{episode}_avg_{avg_reward:.2f}.pkl"
            with open(avg_path, 'wb') as f:
                pickle.dump(self.q_table, f)
            self.best_avg_model_path = avg_path
            print(f"Новая лучшая модель по среднему: {avg_path} (среднее: {avg_reward:.2f})")

    def log_episode_stats(self, episode, avg_reward, min_reward, max_reward):
        """Логирует статистику эпизода."""
        print(f"Эпизод {episode}: avg reward: {avg_reward:.2f}, "
            f"min: {min_reward:.2f}, max: {max_reward:.2f}, "
            f"epsilon: {self.config['EPSILON']:.3f}, "
            f"Q‑table size: {len(self.q_table)}")

    def should_stop_training(self, progress):
        if len(self.ep_rewards) >= self.config['PROGRESS_WINDOW']:
            current_progress = progress[-1] if isinstance(progress, list) else progress

            if self.aggr_ep_rewards['avg']:  # Проверяем наличие данных
                if (current_progress < self.config['PROGRESS_THRESHOLD'] and
                    self.aggr_ep_rewards['avg'][-1] >= self.config['TARGET_REWARD']):
                    print(f"Цель достигнута на эпизоде {len(self.ep_rewards)}! Обучение завершено.")
                    return True

                if (self.config['EPSILON'] <= self.config['MIN_EPSILON'] + 1e-6 and
                    current_progress < self.config['PROGRESS_THRESHOLD'] / 2 and
                    len(self.ep_rewards) > self.config['START_EPSILON_DECAYING'] + self.config['PROGRESS_WINDOW']):
                    print(f"Обучение стабилизировалось на эпизоде {len(self.ep_rewards)}. Остановка.")
                    return True
        return False

    def plot_training_results(self):
        plt.figure(figsize=(15, 16))

        # График 1: Награды по эпизодам
        plt.subplot(4, 1, 1)
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['avg'],
                label='Среднее (скользящее)', color='blue', linewidth=2)
        plt.fill_between(
            self.aggr_ep_rewards['ep'],
            self.aggr_ep_rewards['min'],
            self.aggr_ep_rewards['max'],
            alpha=0.3,
            label='Диапазон (min-max)',
            color='blue'
        )
        plt.axhline(y=self.config['TARGET_REWARD'], color='red',
                linestyle='--', linewidth=2, label='Целевая награда')
        plt.title('Обучение: награды по эпизодам', fontsize=14, fontweight='bold')
        plt.ylabel('Награда', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # График 2: Эпсилон по эпизодам
        plt.subplot(4, 1, 2)
        plt.plot(range(len(self.epsilons_history)), self.epsilons_history,
                color='green', linewidth=2)
        plt.title('Эпсилон по эпизодам (экспоненциальное затухание)',
                fontsize=14, fontweight='bold')
        plt.xlabel('Эпизод', fontsize=12)
        plt.ylabel('Эпсилон', fontsize=12)
        plt.grid(True, alpha=0.3)

        # График 3: Размер Q‑таблицы
        plt.subplot(4, 1, 3)
        episodes_for_stats = self.aggr_ep_rewards['ep']
        table_sizes = [len(self.q_table) for _ in episodes_for_stats]

        plt.plot(episodes_for_stats, table_sizes, color='purple', linewidth=2)
        plt.axhline(y=self.config['PRUNE_THRESHOLD'], color='orange',
                linestyle=':', linewidth=1.5, label='Порог очистки')
        plt.axhline(y=self.config['MAX_Q_TABLE_SIZE'], color='red',
                linestyle='-', linewidth=1.5, label='Максимум')
        plt.title('Размер Q‑таблицы (число посещённых состояний)',
                fontsize=14, fontweight='bold')
        plt.xlabel('Эпизод', fontsize=12)
        plt.ylabel('Число записей', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # График 4: Прогресс обучения
        plt.subplot(4, 1, 4)
        progress_values = self.calculate_progress(
            self.ep_rewards,
            self.config['PROGRESS_WINDOW']
        )

        plt.plot(range(len(progress_values)), progress_values,
                color='orange', linewidth=2, label='Наклон тренда')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5,
                label='Нулевой прогресс')
        plt.axhline(y=self.config['PROGRESS_THRESHOLD'], color='red',
                linestyle='--', alpha=0.8, linewidth=1.5,
                label='Порог остановки')
        plt.title('Прогресс обучения (наклон тренда скользящего среднего)',
                fontsize=14, fontweight='bold')
        plt.xlabel('Эпизод', fontsize=12)
        plt.ylabel('Прогресс (наклон)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.data_path / 'training_results.png', dpi=300, bbox_inches='tight')
        print(f"Графики сохранены: {self.data_path / 'training_results.png'}")
        plt.show()

    def train(self):
        """Основной цикл обучения с сохранением моделей только по лучшему среднему результату."""
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

            # СОХРАНЕНИЕ ПО РАСПИСАНИЮ (контрольные точки) — только по лучшему среднему
            if episode % self.config['SAVE_MODEL_EVERY'] == 0:
                if self.aggr_ep_rewards['avg']:  # Проверяем, есть ли накопленные данные
                    avg_reward_scheduled = self.aggr_ep_rewards['avg'][-1]

                    # Сохраняем только если среднее лучше текущего лучшего
                    if avg_reward_scheduled > self.best_avg_reward:
                        print(f"Сохранение по расписанию (среднее): эпизод {episode}, "
                    f"средняя награда {avg_reward_scheduled:.2f}")
                        self.save_top_models(episode, avg_reward_scheduled)  # Вызов внутри блока if

            # Проверка условий остановки
            progress = self.calculate_progress(self.ep_rewards, self.config['PROGRESS_WINDOW'])
            if self.should_stop_training(progress):
                break

        # Построение графиков после завершения обучения
        self.plot_training_results()


    def _get_discrete_state_linear(self, state):
        """
        Линейная дискретизация: прямое линейное отображение в диапазон дискретных значений.
        Гарантирует равномерное распределение ячеек по диапазону наблюдений.
        """
        discrete_state = []
        for i, val in enumerate(state):
            # Линейное отображение в диапазон дискретных значений
            discrete_val = int(
                (val - self.observation_low[i]) /
                (self.observation_high[i] - self.observation_low[i]) *
                self.config['DISCRETE_OS_SIZE'][i]
            )
            # Жёсткие границы: не выходим за пределы допустимого диапазона
            discrete_val = max(0, min(discrete_val, self.config['DISCRETE_OS_SIZE'][i] - 1))
            discrete_state.append(discrete_val)
        return tuple(discrete_state)


# Запуск обучения
if __name__ == '__main__':
    trainer = QLearningTrainer(CONFIG)

    # Пытаемся загрузить лучшую модель по среднему результату
    if trainer.load_best_average_model():
        print("Обучение продолжено с лучшей сохранённой модели (по среднему).")
    else:
        print("Начинаем новое обучение.")

    trainer.train()
