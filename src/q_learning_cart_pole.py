import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# КОНСТАНТЫ ОБУЧЕНИЯ И НАСТРОЙКИ
ENVIRONMENT_ID = 'CartPole-v1'
TARGET_EPISODE_REWARD = 450
MAX_EPISODES = 3000
CONSECUTIVE_SUCCESS_THRESHOLD = 5

# ПАРАМЕТРЫ Q‑ОБУЧЕНИЯ
DISCOUNT_FACTOR_GAMMA = 0.99

# КОНФИГУРАЦИЯ ДИСКРЕТИЗАЦИИ СОСТОЯНИЙ
STATE_BUCKET_SIZES = [6, 6, 15, 12]
STATE_VALUE_BOUNDS = [
    (-4.8, 4.8),
    (-3.5, 3.5),
    (-0.21, 0.21),
    (-2.0, 2.0)
]

# ОГРАНИЧЕНИЯ И РАЗМЕРЫ
MAX_Q_TABLE_SIZE = 1_000_000
REWARD_HISTORY_WINDOW = 50
PROGRESS_REPORT_FREQUENCY = 100

# ПАРАМЕТРЫ EPSILON‑GREEDY
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1200

# ПАРАМЕТРЫ СКОРОСТИ ОБУЧЕНИЯ
ALPHA_START = 0.3
ALPHA_END = 0.05
ALPHA_DECAY_RATE = 0.002

# НАГРАДЫ И ШТРАФЫ
ANGLE_STABILITY_BONUS_WEIGHT = 2.0
POSITION_CENTERING_BONUS_WEIGHT = 1.0
VELOCITY_PENALTY_WEIGHT = -0.8

# ПАРАМЕТРЫ СГЛАЖИВАНИЯ
EWMA_SMOOTHING_ALPHA = 0.1
EWMA_WARMUP_PERIOD = 5


def check_q_table_size(bucket_sizes, n_actions, max_size):
    """Проверяет, не превышает ли размер Q‑таблицы допустимый лимит."""
    total_size = np.prod(bucket_sizes) * n_actions
    is_acceptable = total_size <= max_size
    return is_acceptable, total_size

def discretize_state_vector(state: np.ndarray) -> tuple:
    """Дискретизирует непрерывное состояние в индексы корзин."""
    discretized = []
    for i, state_value in enumerate(state):
        # Ограничение значений границами
        bounded_value = np.clip(
            state_value,
            STATE_VALUE_BOUNDS[i][0],
            STATE_VALUE_BOUNDS[i][1]
        )
        # Нормализация и дискретизация
        scale = (bounded_value - STATE_VALUE_BOUNDS[i][0]) / \
                (STATE_VALUE_BOUNDS[i][1] - STATE_VALUE_BOUNDS[i][0])
        bucket_idx = int(np.floor(scale * STATE_BUCKET_SIZES[i]))
        # Финальное ограничение индекса
        bucket_idx = np.clip(bucket_idx, 0, STATE_BUCKET_SIZES[i] - 1)
        discretized.append(bucket_idx)
    return tuple(discretized)

def calculate_epsilon(episode: int) -> float:
    """Рассчитывает текущую степень исследования (epsilon)."""
    decay_factor = np.exp(-episode / EPSILON_DECAY)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * decay_factor

def calculate_base_alpha(episode: int) -> float:
    """Базовая скорость обучения с экспоненциальным затуханием."""
    decay_factor = np.exp(-ALPHA_DECAY_RATE * episode)
    return ALPHA_END + (ALPHA_START - ALPHA_END) * decay_factor

def calculate_adaptive_alpha(episode: int, td_error: float = None) -> float:
    """Рассчитывает адаптивную скорость обучения (alpha) с учётом TD‑ошибки."""
    base_alpha = calculate_base_alpha(episode)
    if td_error is None:
        return base_alpha

    # Клиппинг TD‑ошибки в диапазоне [-10, 10]
    clipped_td_error = np.clip(td_error, -10, 10)


    if abs(clipped_td_error) < 1.0:
        alpha = base_alpha * 0.8
    elif abs(clipped_td_error) > 5.0:
        alpha = base_alpha * 1.2
    else:
        alpha = base_alpha
    return max(ALPHA_END, alpha)

def enhanced_reward_function(state: np.ndarray, base_reward: float) -> float:
    """Улучшенная функция награды с бонусами за стабильность."""
    x, x_dot, theta, theta_dot = state
    # Бонус за угол (стабильность)
    angle_bonus = ANGLE_STABILITY_BONUS_WEIGHT * (1 - (abs(theta) / 0.21) ** 2)
    # Бонус за центрирование позиции
    position_bonus = POSITION_CENTERING_BONUS_WEIGHT * (1 - (abs(x) / 4.8) ** 2)
    # Штраф за угловую скорость
    velocity_penalty = VELOCITY_PENALTY_WEIGHT * (abs(theta_dot) / 2.0)
    # Суммарный бонус (не может быть отрицательным)
    total_bonus = max(0, angle_bonus + position_bonus + velocity_penalty)
    return base_reward + total_bonus

def calculate_ewma(scores, alpha=EWMA_SMOOTHING_ALPHA, warmup_period=EWMA_WARMUP_PERIOD):
    """
    Рассчитывает экспоненциально взвешенное скользящее среднее с периодом разогрева.
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

def train_q_learning() -> dict:
    """Основной интерфейс для запуска обучения."""
    try:
        trainer = QLearningTrainer(ENVIRONMENT_ID)
        results = trainer.train()
        return results
    except MemoryError as e:
        print(f"❌ Ошибка: {e}")
        return {}
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return {}

def test_trained_model(q_table: np.ndarray, env_id: str = ENVIRONMENT_ID, n_episodes: int = 10):
    """Тестирование обученной модели на новых эпизодах."""
    env = gym.make(env_id, render_mode=None)
    test_results = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        discretized_state = discretize_state_vector(state)
        total_reward = 0
        steps = 0

        print(f"\n--- Тестовый эпизод {episode + 1} ---")

        while True:
            env.render()
            # Выбор действия (только эксплуатация)
            action = np.argmax(q_table[discretized_state])
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            discretized_state = discretize_state_vector(next_state)
            total_reward += reward
            steps += 1

            if done:
                print(f"  Шагов: {steps}, Reward: {total_reward}")
                test_results.append({'steps': steps, 'reward': total_reward})
                break

    env.close()

    # Статистика тестирования
    avg_steps = np.mean([r['steps'] for r in test_results])
    avg_reward = np.mean([r['reward'] for r in test_results])

    print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Среднее количество шагов: {avg_steps:.2f}")
    print(f"Средний reward: {avg_reward:.2f}")

    return test_results

def analyze_td_errors_distribution(td_errors_history, bins=50):
    """Анализ распределения TD‑ошибок."""
    # Проверка на пустой массив
    if not td_errors_history:
        print("❌ Нет данных для анализа TD‑ошибок")
        return np.array([])

    # Преобразуем в numpy array — работает и со списком чисел, и с массивами
    try:
        all_errors = np.array(td_errors_history)
        # Если получился многомерный массив, сплющим его
        if all_errors.ndim > 1:
            all_errors = all_errors.flatten()
    except Exception as e:
        print(f"❌ Ошибка преобразования данных: {e}")
        return np.array([])

    plt.figure(figsize=(12, 4))

    # Гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(all_errors, bins=bins, alpha=0.7, color='skyblue')
    plt.xlabel('|TD‑ошибка|')
    plt.ylabel('Частота')
    plt.title('Распределение TD‑ошибок')
    plt.grid(True, alpha=0.3)

    # Статистика
    print("Статистика TD‑ошибок:")
    print(f"  Количество ошибок: {len(all_errors):,}")
    print(f"  Среднее: {np.mean(all_errors):.3f}")
    print(f"  Медиана: {np.median(all_errors):.3f}")
    print(f"  Стандартное отклонение: {np.std(all_errors):.3f}")
    print(f"  Минимум: {np.min(all_errors):.3f}")
    print(f"  Максимум: {np.max(all_errors):.3f}")

    # Проверяем, достаточно ли данных для перцентилей
    if len(all_errors) >= 20:
        print(f"  25‑й перцентиль: {np.percentile(all_errors, 25):.3f}")
        print(f"  75‑й перцентиль: {np.percentile(all_errors, 75):.3f}")
        print(f"  95‑й перцентиль: {np.percentile(all_errors, 95):.3f}")
    else:
        print("  ⚠️  Недостаточно данных для расчёта перцентилей")

    # Кумулятивная функция распределения
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(all_errors)
    cumulative = np.arange(len(sorted_errors)) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, color='purple')
    plt.xlabel('|TD‑ошибка|')
    plt.ylabel('Кумулятивная вероятность')
    plt.title('CDF TD‑ошибок')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_errors


class QLearningTrainer:
    def __init__(self, env_id: str = ENVIRONMENT_ID):
        self.env = gym.make(env_id)
        self.q_table = self._initialize_q_table()
        self._initialize_training_history()

    def _initialize_q_table(self) -> np.ndarray:
        """Инициализация Q‑таблицы с проверкой размера."""
        is_valid, size = check_q_table_size(
            STATE_BUCKET_SIZES,
            self.env.action_space.n,
            MAX_Q_TABLE_SIZE
        )
        if not is_valid:
            raise MemoryError(f"Q‑таблица слишком большая: {size:,} элементов")
        shape = STATE_BUCKET_SIZES + [self.env.action_space.n]
        print(f"✅ Q‑таблица создана: {size:,} элементов")
        return np.zeros(shape)

    def _initialize_training_history(self):
        """Инициализация истории обучения."""
        self.scores = []
        self.avg_scores = []
        self.td_errors_history = []
        self.epsilon_history = []
        self.alpha_history = []
        self.best_q_table = None
        self.best_avg_reward = None

    def train(self) -> dict:
        """Основной цикл обучения."""
        for episode in range(MAX_EPISODES):
            episode_result = self._run_episode(episode)
            self._update_training_history(episode_result, episode)
            if self._check_success_condition():
                break
            # Логирование прогресса
            if (episode + 1) % PROGRESS_REPORT_FREQUENCY == 0:
                self._log_progress(episode + 1, episode_result)
        return self._get_training_results()

    def _run_episode(self, episode: int) -> dict:
        """Запуск одного эпизода обучения."""
        state, _ = self.env.reset()
        discretized_state = discretize_state_vector(state)
        score = 0
        td_errors = []
        epsilon = calculate_epsilon(episode)

        for t in range(500):  # Максимум шагов в эпизоде
            # Выбор действия: исследование или эксплуатация
            action = self._select_action(discretized_state, epsilon)
            # Взаимодействие с окружением
            next_state, base_reward, done, truncated, _ = self.env.step(action)
            done = done or truncated
            # Расчёт улучшенной награды
            reward = enhanced_reward_function(next_state, base_reward)
            # Дискретизация следующего состояния
            discretized_next_state = discretize_state_vector(next_state)

            # Расчёт TD‑ошибки
            best_next_action = np.argmax(self.q_table[discretized_next_state])
            td_target = reward + DISCOUNT_FACTOR_GAMMA * self.q_table[discretized_next_state][best_next_action]
            td_error = td_target - self.q_table[discretized_state + (action,)]
            td_errors.append(abs(td_error))

            # Адаптивная скорость обучения
            alpha = calculate_adaptive_alpha(episode, td_error)

            # Обновление Q‑таблицы
            indices = discretized_state + (action,)
            self.q_table[indices] += alpha * td_error

            # Переход к следующему состоянию
            discretized_state = discretized_next_state
            score += base_reward

            if done:
                break

        return {
            'score': score,
            'td_errors': td_errors,
            'epsilon': epsilon,
            'alpha': alpha
        }

    def _select_action(self, state: tuple, epsilon: float) -> int:
        """Выбор действия по стратегии epsilon‑greedy."""
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def _update_training_history(self, episode_result: dict, episode: int):
        """Обновление истории обучения после эпизода."""
        self.scores.append(episode_result['score'])

        # Расчёт среднего reward за последние N эпизодов
        if len(self.scores) >= REWARD_HISTORY_WINDOW:
            avg_score = np.mean(self.scores[-REWARD_HISTORY_WINDOW:])
        else:
            avg_score = np.mean(self.scores)
        self.avg_scores.append(avg_score)

        # СОХРАНЯЕМ ВСЕ TD‑ОШИБКИ ЗА ЭПИЗОД (по шагам)
        td_errors = episode_result.get('td_errors', [])
        if td_errors:  # Проверяем, что ошибки есть
            self.td_errors_history.extend(td_errors)  # Используем extend, а не append

        # Сохранение параметров
        self.epsilon_history.append(episode_result['epsilon'])
        self.alpha_history.append(episode_result['alpha'])

        # Обновление лучшей модели
        if self.best_avg_reward is None or avg_score > self.best_avg_reward:
            self.best_avg_reward = avg_score
            self.best_q_table = self.q_table.copy()
            print(f"  🏆 ОБНОВЛЕНИЕ ЛУЧШЕЙ МОДЕЛИ: средний reward = {self.best_avg_reward:.2f}")

    def _check_success_condition(self) -> bool:
        """Проверка условия стабильного успеха."""
        if len(self.avg_scores) < CONSECUTIVE_SUCCESS_THRESHOLD:
            return False

        recent_avg_scores = self.avg_scores[-CONSECUTIVE_SUCCESS_THRESHOLD:]
        return all(avg >= TARGET_EPISODE_REWARD for avg in recent_avg_scores)

    def _log_progress(self, current_episode: int, episode_result: dict):
        """Логирование прогресса обучения."""
        ewma_score = calculate_ewma(self.scores[-REWARD_HISTORY_WINDOW:], warmup_period=EWMA_WARMUP_PERIOD)
        print(f"\n--- Эпизод {current_episode} ---")
        print(f"  EWMA reward: {ewma_score:.2f}")
        print(f"  Средний reward ({REWARD_HISTORY_WINDOW} эпизодов): {self.avg_scores[-1]:.2f}")
        print(f"  Текущий epsilon: {episode_result['epsilon']:.3f}")
        print(f"  Текущая alpha: {episode_result['alpha']:.3f}")
        print(f"  Средняя |TD‑ошибка|: {self.td_errors_history[-1]:.3f}")
        if self._check_success_condition():
            print(f"✅ СТАБИЛЬНЫЙ УСПЕХ ДОСТИГНУТ: {CONSECUTIVE_SUCCESS_THRESHOLD} эпизодов подряд ≥ {TARGET_EPISODE_REWARD}")


    def _get_training_results(self) -> dict:
        """Получение результатов обучения."""
        return {
            'scores': self.scores,
            'avg_scores': self.avg_scores,
            'td_errors_history': self.td_errors_history,
            'best_q_table': self.best_q_table,
            'best_avg_reward': self.best_avg_reward,
            'epsilon_history': self.epsilon_history,
            'alpha_history': self.alpha_history
        }



class TrainingVisualizer:
    @staticmethod
    def plot_training_results(results: dict):
        """Построение всех графиков результатов обучения."""
        plt.figure(figsize=(15, 10))

        # График 1: Reward за эпизод и средний reward
        plt.subplot(2, 2, 1)
        plt.plot(results['scores'], label='Reward за эпизод', alpha=0.6)
        if len(results['avg_scores']) > 0:
            episodes_x = list(range(REWARD_HISTORY_WINDOW - 1,
                              REWARD_HISTORY_WINDOW - 1 + len(results['avg_scores'])))
            plt.plot(episodes_x, results['avg_scores'],
                    label=f'Средний reward ({REWARD_HISTORY_WINDOW} эпизодов)',
            color='red')
        plt.axhline(y=TARGET_EPISODE_REWARD, color='green', linestyle='--',
                label=f'Целевой reward {TARGET_EPISODE_REWARD}')
        plt.xlabel('Эпизод')
        plt.ylabel('Reward')
        plt.title('Обучение Q‑learning: Reward динамика')
        plt.legend()
        plt.grid(True)

        # График 2: TD‑ошибки
        plt.subplot(2, 2, 2)
        plt.plot(results['td_errors_history'], color='orange')
        plt.xlabel('Эпизод')
        plt.ylabel('Средняя |TD‑ошибка|')
        plt.title('Динамика TD‑ошибок')
        plt.grid(True)

        # График 3: Эволюция epsilon
        plt.subplot(2, 2, 3)
        plt.plot(results['epsilon_history'], color='purple')
        plt.xlabel('Эпизод')
        plt.ylabel('Epsilon')
        plt.title('Эволюция степени исследования (epsilon)')
        plt.grid(True)

        # График 4: Эволюция alpha
        plt.subplot(2, 2, 4)
        plt.plot(results['alpha_history'], color='brown')
        plt.xlabel('Эпизод')
        plt.ylabel('Alpha')
        plt.title('Эволюция скорости обучения (alpha)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_q_table_slice(q_table: np.ndarray, state_slice: tuple = None, action_to_show: int = 0):
        """Визуализация среза Q‑таблицы для анализа (одно действие)."""
        if state_slice is None:
            state_slice = (slice(None), slice(None), 0, 0)

        # Берём срез только для указанного действия
        full_slice = state_slice + (action_to_show,)
        q_slice_2d = q_table[full_slice]  # Теперь форма (6, 6)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(q_slice_2d, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Q‑значение')
        plt.xlabel('Индекс состояния 1')
        plt.ylabel('Индекс состояния 2')
        plt.title(f'Срез Q‑таблицы: Q(s, a={action_to_show})')
        plt.show()


def main():
    """Основной скрипт запуска обучения и визуализации."""
    print("🚀 ЗАПУСК ОБУЧЕНИЯ Q‑LEARNING")
    print("=" * 50)

    # Обучение модели
    results = train_q_learning()

    if not results:
        print("❌ Обучение не завершено из‑за ошибки")
        return

    print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Лучший средний reward: {results['best_avg_reward']:.2f}")

    # АНАЛИЗ РАСПРЕДЕЛЕНИЯ TD‑ОШИБОК
    print("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ TD‑ОШИБОК...")
    all_errors = analyze_td_errors_distribution(results['td_errors_history'])

    # Визуализация результатов
    print("\n📊 ПОСТРОЕНИЕ ГРАФИКОВ...")
    visualizer = TrainingVisualizer()
    visualizer.plot_training_results(results)

    # Визуализация Q‑таблицы
    print("\n🔎 АНАЛИЗ Q‑ТАБЛИЦЫ...")
    visualizer.visualize_q_table_slice(results['best_q_table'])

    # Тестирование модели
    print("\n🧪 ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ...")
    test_results = test_trained_model(results['best_q_table'])



if __name__ == "__main__":
    main()   