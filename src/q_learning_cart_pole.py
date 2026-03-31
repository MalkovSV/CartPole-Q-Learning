import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from functools import lru_cache
import heapq
import time

# КОНФИГУРАЦИЯ
CONFIG = {
    # Основные параметры обучения
    'EPISODES': 1800,
    'RENDER_EVERY': 20,

    # Параметры Q‑learning
    'LEARNING_RATE': 0.01,
    'DISCOUNT': 0.99,

    # ε‑жадная стратегия
    'EPSILON': 1.0,
    'START_EPSILON_DECAYING': 100,      # Начинаем затухание после 100 эпизодов
    'MIN_EPSILON': 0.05,             # Минимальное значение эпсилона
    'EPSILON_DECAY_RATE': 0.002,   # Коэффициент экспоненциального затухания (λ)

    # Дискретизация пространства состояний
    'DISCRETE_OS_SIZE': [20] * 4,  # CartPole имеет 4 измерения состояния
    'DISCRETIZATION_METHOD': 'linear',  # 'linear' или 'sigmoid'


    # Целевые показатели
    'TARGET_REWARD': 450,

    # Сохранение моделей
    'SAVE_MODEL_EVERY': 20,

    # ПАРАМЕТРЫ ДЛЯ АНАЛИЗА ПРОГРЕССА
    'PROGRESS_WINDOW': 50,         # окно для анализа прогресса
    'PROGRESS_THRESHOLD': 0.05,    # порог для адаптации

    # ОГРАНИЧЕНИЕ НА РАЗМЕР Q‑ТАБЛИЦЫ
    'PRUNE_THRESHOLD': 4000,        # Запускать очистку при 3000 записях
    'MAX_Q_TABLE_SIZE': 5000,       # Максимальный размер — 4000
    'PRUNE_TARGET_RATIO': 0.75,      # Оставлять 80 % от максимума

    'ENABLE_PRUNE_LOGGING': False,  # По умолчанию — вывод выключён

    # Логирование инициализации Q‑значений
    'ENABLE_INIT_LOGGING': False,  # Включить/выключить логирование новых записей
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
        # Новый атрибут: история размеров Q‑таблицы
        self.q_table_sizes_history = []

        # Оставляем только отслеживание лучшего среднего результата
        self.best_avg_reward = float('-inf')  # Лучшее среднее значение награды
        self.best_avg_model_path = None      # Путь к лучшей модели по среднему

        self.state_visit_count = {}  # Словарь для учёта посещаемости состояний
        self.state_last_visited = {}

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
        key = discrete_state + (action,)
        if key not in self.q_table:
            # Оптимистичная инициализация с небольшим шумом
            optimistic_value = 1.0
            noise = np.random.normal(0, 0.05)  # Небольшой шум
            self.q_table[key] = optimistic_value + noise
            # Гарантируем, что значение остаётся положительным
            self.q_table[key] = max(self.q_table[key], 0.5)

            # Логирование новых записей (опционально, для отладки)
            if self.config.get('ENABLE_INIT_LOGGING', False):
                print(f"Создана новая запись Q‑таблицы: {key} = {self.q_table[key]:.3f}")
        return self.q_table[key]


    def _set_q_value(self, discrete_state, action, value):
        """Устанавливает Q‑значение для состояния и действия с ограничением."""
        key = discrete_state + (action,)

        # Реалистичные границы для CartPole-v1
        MAX_Q = 500  # Максимум за эпизод (500 шагов × награда 1)
        MIN_Q = -10  # Минимальный штраф

        clipped_value = np.clip(value, MIN_Q, MAX_Q)
        self.q_table[key] = clipped_value

    def _get_discrete_state_linear(self, state):
        discrete_state = []
        for i, val in enumerate(state):
            low = self.observation_low[i]
            high = self.observation_high[i]

            # Защита от нулевого диапазона
            if abs(high - low) < 1e-6:
                discrete_val = 0
            else:
                # Нормализация в [0, 1] с жёстким ограничением
                normalized = (val - low) / (high - low)
                normalized = np.clip(normalized, 0.0, 1.0)

                # Дискретизация с floor вместо round
                discrete_val = int(np.floor(normalized * self.config['DISCRETE_OS_SIZE'][i]))
                # Ограничение диапазона: 0 ≤ discrete_val ≤ DISCRETE_OS_SIZE[i] - 1
                discrete_val = max(0, min(discrete_val, self.config['DISCRETE_OS_SIZE'][i] - 1))

            discrete_state.append(discrete_val)
        return tuple(discrete_state)


    def get_discrete_state_sigmoid(self, state_tuple):
        """
        Преобразование непрерывного состояния в дискретное (sigmoid‑метод)
        с улучшенной нормализацией и обработкой бесконечных границ.
        """
        state = np.array(state_tuple, dtype=np.float64)

        if state.size != len(self.OBSERVATION_LOW_TUPLE):
            raise ValueError(
                f"Размер state ({state.size}) не соответствует ожидаемому "
                f"({len(self.OBSERVATION_LOW_TUPLE)})"
            )

        valid_range = np.isfinite(self.observation_high) & np.isfinite(self.observation_low)
        normalized = np.zeros_like(state)

        # Нормализация для конечных границ
        if np.any(valid_range):
            range_values = self.observation_high[valid_range] - self.observation_low[valid_range]
            # Защита от нулевого диапазона
            range_values = np.where(range_values == 0, 1.0, range_values)

            normalized[valid_range] = (
                (state[valid_range] - self.observation_low[valid_range]) / range_values
            )
            # Жёсткое ограничение в [0, 1]
            normalized[valid_range] = np.clip(normalized[valid_range], 0.0, 1.0)

        # Преобразование для бесконечных границ с использованием tanh
        infinite_mask = ~valid_range
        if np.any(infinite_mask):
            # tanh даёт значения в [-1, 1], преобразуем в [0, 1]
            tanh_values = np.tanh(state[infinite_mask] * 0.2)  # Коэффициент 0.2 для сжатия
            normalized[infinite_mask] = (tanh_values + 1) / 2

        # Дискретизация: обрабатываем каждое измерение отдельно
        discrete_state = []
        for i, norm_val in enumerate(normalized):
            size = self.config['DISCRETE_OS_SIZE'][i]
            discrete_val = int(norm_val * (size - 1))
            discrete_val = max(0, min(discrete_val, size - 1))
            discrete_state.append(discrete_val)

        return tuple(discrete_state)

    def calculate_progress(self, rewards, window):
        """
        Рассчитывает прогресс обучения как разницу между средним вознаграждением
        во второй половине окна и первой половине окна.

        Args:
            rewards: список накопленных наград по эпизодам
            window: размер окна для анализа прогресса (должно быть чётным)

        Returns:
            Список значений прогресса той же длины, что и rewards.
            Для первых (window-1) эпизодов прогресс = 0.0.
        """
        # Убедимся, что окно чётное — иначе разделение на половины некорректно
        if window % 2 != 0:
            raise ValueError("window должен быть чётным числом для корректного разделения на половины")

        if len(rewards) < window:
            return [0.0] * len(rewards)

        progress_values = [0.0] * (window - 1)  # Первые (window-1) значений — нули
        half = window // 2
        
        for i in range(window - 1, len(rewards)):
            # Вторая половина: последние 'half' эпизодов в текущем окне
            # Индексы: от (i - half + 1) до i (включительно)
            second_half = rewards[i - half + 1:i + 1]

            # Первая половина: предыдущие 'half' эпизодов перед второй половиной
            # Индексы: от (i - window + 1) до (i - half) (включительно)
            first_half = rewards[i - window + 1:i - half + 1]

            # Вычисляем средние значения, обрабатывая случай пустых списков
            avg_first = np.mean(first_half) if first_half else 0.0
            avg_second = np.mean(second_half) if second_half else 0.0

            # Разница между средними: положительный прогресс — улучшение, отрицательный — ухудшение
            progress = avg_second - avg_first
            progress_values.append(progress)

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

            # Учёт посещаемости состояния
            state_key = discrete_state + (action,)
            if state_key in self.state_visit_count:
                self.state_visit_count[state_key] += 1
            else:
                self.state_visit_count[state_key] = 1 

            self.state_last_visited[state_key] = episode

        env.close()
        return episode_reward

    def update_epsilon(self, episode):
        """
        Обновляет значение эпсилона по экспоненциальному закону.
        После достижения минимального значения эпсилон остаётся постоянным.
        """
        if episode >= self.config['START_EPSILON_DECAYING']:
            decay_steps = episode - self.config['START_EPSILON_DECAYING']
            self.config['EPSILON'] = (
                self.config['MIN_EPSILON'] +
                (1.0 - self.config['MIN_EPSILON']) *
                np.exp(-self.config['EPSILON_DECAY_RATE'] * decay_steps)
            )
            # Гарантируем, что эпсилон не опустится ниже установленного минимума
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
        if len(self.ep_rewards) < self.config['PROGRESS_WINDOW']:
            return False

        current_progress = progress[-1] if isinstance(progress, list) else progress

        if self.aggr_ep_rewards['avg']:
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

    def prune_q_table(self):
        current_size = len(self.q_table)

        # Проверяем, нужно ли выполнять очистку по размеру таблицы
        if current_size <= self.config['PRUNE_THRESHOLD']:
            if self.config.get('ENABLE_PRUNE_LOGGING', True):
                print(f"Очистка не требуется: текущий размер {current_size} <= порог {self.config['PRUNE_THRESHOLD']}")
            return

        # Рассчитываем прогресс обучения
        progress_values = self.calculate_progress(
            self.ep_rewards,
            self.config['PROGRESS_WINDOW']
        )

        # Если недостаточно данных для расчёта прогресса или прогресс положительный, пропускаем очистку
        if (len(progress_values) < self.config['PROGRESS_WINDOW'] or
                progress_values[-1] > self.config['PROGRESS_THRESHOLD']):
            if self.config.get('ENABLE_PRUNE_LOGGING', True):
                print(f"Пропуск очистки: текущий прогресс {progress_values[-1]:.3f} "
                    f"> порог {self.config['PROGRESS_THRESHOLD']}. Обучение прогрессирует.")
            return

        # Рассчитываем целевой размер после очистки
        target_size = int(self.config['MAX_Q_TABLE_SIZE'] * self.config['PRUNE_TARGET_RATIO'])
        to_remove_total = max(0, current_size - target_size)

        if to_remove_total <= 0:
            return

        if self.config.get('ENABLE_PRUNE_LOGGING', True):
            print(f"Запуск очистки Q‑таблицы: нужно удалить {to_remove_total} записей из {current_size}")
            print(f"Целевой размер после очистки: {target_size}")

        # Создаём список записей с их приоритетами сохранения
        records_with_priority = []
        current_episode = len(self.ep_rewards)

        for key in self.q_table.keys():
            visit_count = self.state_visit_count.get(key, 1)
            q_value = self.q_table[key]
            last_visited = self.state_last_visited.get(key, 0)

            # Приоритет сохранения (чем выше, тем важнее сохранить)
            preservation_priority = (
                np.log(visit_count) * 0.3 +  # Логарифм посещаемости
                q_value * 0.6 +             # Вес Q‑значения
                (current_episode - last_visited) * (-0.1)  # Время последнего посещения
            )
            records_with_priority.append((preservation_priority, key, visit_count, q_value, last_visited))

        # Сортируем по приоритету сохранения (убывание) — наименее важные в конце
        records_with_priority.sort(key=lambda x: x[0], reverse=True)

        # Определяем ключи для удаления: последние N записей в отсортированном списке
        keys_to_remove = [record[1] for record in records_with_priority[-to_remove_total:]]

        # ВЫВОД ЗАПИСЕЙ, КОТОРЫЕ БУДУТ УДАЛЕНЫ
        if self.config.get('ENABLE_PRUNE_LOGGING', True):
            print("\n" + "=" * 90)
            print("ЗАПИСИ, ПОДЛЕЖАЩИЕ УДАЛЕНИЮ (сортировка по приоритету удаления — от наименее важных):")
            print("=" * 90)
            print(f"{'№':<3} {'Состояние':<20} {'Действие':<8} {'Q‑value':<10} {'Посещ.':<8} {'Послед. посещ.':<12} {'Приоритет':<10}")
            print("-" * 90)

            removed_count = 0
            total_visit = 0.0
            total_q = 0.0
            total_priority = 0.0

            # Показываем до 30 записей для обзора (или все, если их меньше)
            show_limit = min(100, len(keys_to_remove))
            for i, key in enumerate(keys_to_remove[:show_limit], 1):
                state_action = key[:-1]
                action = key[-1]
                visit_count = self.state_visit_count.get(key, 0)
                q_value = self.q_table.get(key, 0.0)
                last_visited = self.state_last_visited.get(key, 0)

                # Находим приоритет этой записи для вывода
                priority = next(
                    record[0] for record in records_with_priority
                    if record[1] == key
                )

                print(f"{i:<3} {str(state_action):<20} {action:<8} {q_value:<10.4f} {visit_count:<8} {last_visited:<12} {priority:<10.4f}")


                total_visit += visit_count
                total_q += q_value
                total_priority += priority
                removed_count += 1

            if len(keys_to_remove) > show_limit:
                print(f"... и ещё {len(keys_to_remove) - show_limit} записей.")

            print("-" * 90)

            # Статистика по удаляемым записям
            avg_visit = total_visit / removed_count if removed_count > 0 else 0
            avg_q = total_q / removed_count if removed_count > 0 else 0
            avg_priority = total_priority / removed_count if removed_count > 0 else 0

            print(f"Статистика по удаляемым: средняя посещаемость = {avg_visit:.1f}, "
                f"средний Q‑value = {avg_q:.4f}, средний приоритет = {avg_priority:.4f}")
            print(f"Всего будет удалено: {len(keys_to_remove)} записей")

        # Фактическое удаление записей
        for key in keys_to_remove:
            try:
                del self.q_table[key]
                if key in self.state_visit_count:
                    del self.state_visit_count[key]
                if key in self.state_last_visited:
                    del self.state_last_visited[key]
            except KeyError:
                continue

        if self.config.get('ENABLE_PRUNE_LOGGING', True):
            print(f"Q‑таблица очищена. Новый размер: {len(self.q_table)}")

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

        # График 3: Размер Q‑таблицы — исправленная версия
        plt.subplot(4, 1, 3)
        all_episodes = list(range(len(self.q_table_sizes_history)))
        q_sizes = self.q_table_sizes_history

        plt.plot(all_episodes, q_sizes, color='purple', linewidth=2, alpha=0.8)

        # Создаём корректные данные для scatter-графика: берём эпизоды и соответствующие размеры Q‑таблицы
        scatter_episodes = []
        scatter_sizes = []

        for ep in self.aggr_ep_rewards['ep']:
            if ep < len(self.q_table_sizes_history):  # Проверяем, что эпизод не выходит за границы истории
                scatter_episodes.append(ep)
                scatter_sizes.append(self.q_table_sizes_history[ep])

        # Отрисовываем точки статистики
        if scatter_episodes:  # Проверяем, что есть данные для отображения
            plt.scatter(scatter_episodes, scatter_sizes,
                    color='red', s=50, zorder=5, label='Точки статистики')

        plt.title('Размер Q‑таблицы по эпизодам', fontsize=14, fontweight='bold')
        plt.xlabel('Эпизод', fontsize=12)
        plt.ylabel('Размер Q‑таблицы', fontsize=12)
        plt.legend()
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
        """Основной цикл обучения с сохранением моделей при улучшении среднего результата."""
        for episode in range(self.config['EPISODES']):
            # Обучение одного эпизода
            episode_reward = self.train_episode(episode)
            self.ep_rewards.append(episode_reward)
            self.epsilons_history.append(self.config['EPSILON'])

            # Обновление эпсилона
            self.update_epsilon(episode)

            # Проверка и очистка Q‑таблицы при достижении порога
            if len(self.q_table) > self.config['PRUNE_THRESHOLD']:
                self.prune_q_table()

            # Запись размера Q‑таблицы после каждого эпизода
            self.q_table_sizes_history.append(len(self.q_table))

            # Сбор статистики и сохранение при улучшении каждые RENDER_EVERY эпизодов
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

                # В цикле обучения, после завершения эпизода и расчёта avg_reward
                if avg_reward < 50 and episode % 20 == 0:
                    if self.config.get('ENABLE_PRUNE_LOGGING', True):
                        print(f"\n{'='*120}")
                        print(f"ОТЛАДКА: НИЗКИЙ REWARD НА ЭПИЗОДЕ {episode}")
                        print(f"Средний reward = {avg_reward:.2f}, размер Q‑таблицы = {len(self.q_table)}")
                        print(f"{'='*120}")

                        # Выводим структурированную Q‑таблицу (первые 20 состояний, отсортированных по Q‑value)
                        self.print_q_table_full()

                        print(f"{'-'*120}\n")

                # СОХРАНЕНИЕ ПРИ УЛУЧШЕНИИ СРЕДНЕГО РЕЗУЛЬТАТА
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    avg_path = self.data_path / f"best_avg_q_table_episode_{episode}_avg_{avg_reward:.2f}.pkl"
                    with open(avg_path, 'wb') as f:
                        pickle.dump(self.q_table, f)
                    self.best_avg_model_path = avg_path
                    print(f"✅ Новая лучшая модель: эпизод {episode}, среднее: {avg_reward:.2f}")

            # Логирование прогресса каждые 100 эпизодов
            if episode % 100 == 0:
                # Рассчитываем прогресс с учётом окна анализа
                progress_values = self.calculate_progress(
                    self.ep_rewards,
                    self.config['PROGRESS_WINDOW']
                )

                # Проверяем, достаточно ли данных для расчёта прогресса
                if len(progress_values) >= self.config['PROGRESS_WINDOW']:
                    current_progress = progress_values[-1]
                    progress_status = "ПРОГРЕСС" if current_progress > self.config['PROGRESS_THRESHOLD'] else "СТАБИЛЬНО"

                    print(f"📊 Прогресс на эпизоде {episode}: {current_progress:.3f} "
                        f"({progress_status}) | Порог: {self.config['PROGRESS_THRESHOLD']:.3f}")
                else:
                    # Если данных недостаточно, выводим сообщение
                    remaining_episodes = self.config['PROGRESS_WINDOW'] - len(progress_values)
                    print(f"⏱️  Ожидание данных для расчёта прогресса: "
                        f"нужно ещё {remaining_episodes} эпизодов")

            # Проверка условий остановки
            progress = self.calculate_progress(self.ep_rewards, self.config['PROGRESS_WINDOW'])
            if self.should_stop_training(progress):
                break

        # Построение графиков после завершения обучения
        self.plot_training_results()

    def print_q_table_full(self):
        """Выводит все записи Q‑таблицы с посещаемостью и значениями, отсортированные по Q‑value."""
        if not self.config.get('ENABLE_PRUNE_LOGGING', True):
            return  # Выходим, если вывод отключён

        print("\n" + "=" * 80)
        print("ПОЛНАЯ Q‑ТАБЛИЦА (состояние, действие) → Q‑value, посещаемость")
        print("=" * 80)

        if not self.q_table:
            print("Q‑таблица пуста.")
            return

        # Создаём список кортежей (ключ, Q‑значение, посещаемость) для сортировки
        table_data = []
        for key, q_value in self.q_table.items():
            visit_count = self.state_visit_count.get(key, 0)  # Получаем посещаемость, если есть
            table_data.append((key, q_value, visit_count))

        # Сортируем по убыванию Q‑значения
        sorted_table = sorted(table_data, key=lambda x: x[1], reverse=True)

        # Выводим первые 50 записей (чтобы не перегружать вывод)
        max_display = 20
        count = 0
        for key, q_value, visit_count in sorted_table:
            if count >= max_display:
                break
            state_action = key[:-1]  # Состояние (все элементы кроме последнего)
            action = key[-1]          # Действие (последний элемент)
            print(f"Состояние: {state_action}, Действие: {action} → Q‑value: {q_value:.4f}, Посещаемость: {visit_count}")
            count += 1

        # Если записей больше, чем выведено, сообщаем об этом
        if len(sorted_table) > max_display:
            print(f"\n... и ещё {len(sorted_table) - max_display} записей (всего: {len(sorted_table)})")

        print(f"\nВсего записей в Q‑таблице: {len(self.q_table)}")
        print(f"Уникальных состояний: {len(set(key[:-1] for key in self.q_table.keys()))}")
        print(f"Среднее Q‑значение: {np.mean([v for v in self.q_table.values()]):.4f}")
        print(f"Максимальное Q‑значение: {max(self.q_table.values()):.4f}")
        print(f"Минимальное Q‑значение: {min(self.q_table.values()):.4f}")

    import time

    def real_time_visualization(self, num_episodes=2, delay=0.02):
        """Просмотр работы лучшей модели в реальном времени с задержкой для лучшей визуализации."""
        print(f"\n{'='*50}")
        print("ДЕМОНСТРАЦИЯ РАБОТЫ ЛУЧШЕЙ МОДЕЛИ")
        print(f"{'='*50}")

        # Загружаем лучшую модель, если она есть
        if self.best_avg_model_path and self.load_best_average_model():
            print(f"Загружена лучшая модель: {self.best_avg_model_path.name}")
        else:
            print("Нет сохранённой лучшей модели. Используем текущую Q‑таблицу.")

        env = gym.make('CartPole-v1', render_mode='human')

        try:
            for episode in range(num_episodes):
                print(f"\n=== ЭПИЗОД {episode + 1} из {num_episodes} ===")
                state, _ = env.reset()
                discrete_state = self.get_discrete_state(tuple(state))
                total_reward = 0
                step_count = 0

                # Ограничение по количеству шагов для предотвращения бесконечного цикла
                max_steps = 500

                while step_count < max_steps:
                    # Выбираем действие — только эксплуатация (без случайного выбора)
                    q_values = [
                        self._get_q_value(discrete_state, a)
                for a in range(self.action_space_n)
                    ]
                    action = np.argmax(q_values)

                    # Выполняем действие
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Обновляем состояние *до* обновления статистики
                    discrete_state = self.get_discrete_state(tuple(next_state))

                    total_reward += reward
                    step_count += 1

                    # Дополнительная информация для отладки (можно отключить)
                    print(f"Шаг {step_count}: действие={action}, награда={reward:.1f}, "
                        f"Q‑значения={q_values}, состояние={discrete_state}")

                    # Задержка для лучшей визуализации
                    time.sleep(delay)

                    if done:
                        print(f"Эпизод завершён. Reward: {total_reward}, Шагов: {step_count}")
                        break

                if step_count >= max_steps:
                    print(f"Эпизод прерван: достигнуто максимальное число шагов ({max_steps}).")

        except Exception as e:
            print(f"Ошибка во время демонстрации: {e}")
        finally:
            env.close()
            print(f"\nДемонстрация завершена.")

# Запуск обучения
if __name__ == '__main__':
    trainer = QLearningTrainer(CONFIG)

    # Пытаемся загрузить лучшую модель по среднему результату
    if trainer.load_best_average_model():
        print("Обучение продолжено с лучшей сохранённой модели (по среднему).")
    else:
        print("Начинаем новое обучение.")

    trainer.train()

    # Запуск демонстрации работы лучшей модели после обучения
    trainer.real_time_visualization(num_episodes=3)