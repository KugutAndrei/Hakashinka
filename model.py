# from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# class LossOptimizer(ABC):
#     """
#     Этот класс находит наилучшие параметры для оптимизации функции потерь
#     """
#     def __init__(self,
#                  min_params =[0, 10],
#                  max_params =[0, 10],
#                  param_steps=[1, 1]
#     ):
#         self.min_params = min_params
#         self.max_params = max_params
#         self.param_steps = param_steps

#     @abstractmethod
#     def fit(self, loss_func):
#         pass

#     @abstractmethod
#     def predict(self, value):
#         pass


# optimization?
class OptimalDist():
    """
    Бесполезный кусок говна, который просто жрет время
    """
    def __init__(self, grid_params:any = [(0, 10, 1),
                                      (50, 100, 50)]
    ):
        self.hasFit = False
        self.dist = grid_params[0]
        self.cars = grid_params[1]

    # def __init__(self, n_estimators:int = 100, max_depth: int = 3):
    #     self.n_estimators = n_estimators
    #     self.max_depth = max_depth
    #     self.hasFit = False

    def fit(self, loss_func:callable, n_estimators:int=100, max_depth:int=3) -> None:
        """
        Обучает модель
        """
        X = pd.DataFrame({"n_cars": []})
        y = pd.Series(name="best_dist")

        for n_cars in range(*self.cars):
            max_loss = -1.0
            best_dist = -1.0
            for dist in np.arange(*self.dist):
                loss_res = loss_func(dist, n_cars)
                if loss_res > max_loss:
                    max_loss = loss_res
                    best_dist = dist
                X.loc[len(X)] = n_cars
                y.loc[len(y)] = best_dist
        
        self.reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.reg.fit(X, y)
        self.hasFit = True
        print(X)
        print(y)
        return self.reg

    def predict(self, n_cars:int | np.ndarray) -> float | np.ndarray:
        """
        Предсказывает наилучшую дистанцию для заданного числа машин
        """
        assert self.hasFit, "You should fit model first"
        y = np.array(n_cars).reshape(-1, 1)
        y = y.reshape(len(y), -1)
        return self.reg.predict(y)


class OptimalLoss_Q():
    """
    Ищет argmax(Loss(x, y), y = y_0) c помощью обучения с подкреплением
    """
    def __init__(self, grid_params = [(0, 10, 1),
                                      (0, 10, 1)]
    ):
        self.grid_params= grid_params


    def fit(self, loss_func: callable,
            alpha: float = 0.1,  # шаг обучения (learning rate)
            gamma: float = 0.9,  # дисконт-фактор
            epsilon: float=0.1,  # эпсилон для epsilon-greedy стратегии
            max_step_in_episode: int = 100,
            num_episodes: int = 1000,  # количество эпох
            showInfo: bool = False  # покзаывает результаты обучения после каждой эпохи
    ):
        self.loss_func = loss_func

        # Инициализация Q-таблицы
        self.action_values = np.arange(*self.grid_params[0])  # возможные действия (дистанция)
        self.state_values = np.arange(*self.grid_params[1])   # возможные состояния (поток машин)
        n_actions = len(self.action_values)
        n_states = len(self.state_values)
        
        self.Q = np.zeros((n_states, n_actions))

        def get_new_state(current_state, action):
            new_state = (current_state + action) % n_states
            return new_state

        # Тренировка модели  
        for episode in range(num_episodes):
            # В начале каждой эпохи начальные параметры
            start_state = n_states // 2
            state = start_state
            total_reward = 0
            
            for i in range(max_step_in_episode):
                # Выбор действия (dist) с помощью epsilon-greedy стратегии
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(self.Q[state, :])
                
                # Выполнение действия
                x = self.action_values[action]
                y = self.state_values[state]
                reward = -self.loss_func(x, y)  # награда -значение функции потерь
                
                # Вычисление нового состояния
                new_state = get_new_state(state, action)  # формируем новое значение n
                
                # Обновление Q-таблицы
                self.Q[state, action] = (1 - alpha) * self.Q[state, action] + alpha * (reward + gamma * np.max(self.Q[new_state, :]))
                
                # Переход к новому состоянию
                state = new_state
                total_reward += reward
            
            if showInfo:
                print(f"Episode {episode}, Total Reward: {total_reward}")


    def _find_ind_(self, arr :np.ndarray, x: int|float, lineApprox: bool = True):
        abs_diff = np.abs(arr - x)
        ind = np.where(abs_diff == 0)
        if len(ind[0] == 1):  # попали в узел сетки
            return [(ind[0][0], 1)]
        # не попали в узел сетки
        closest_inds = np.argsort(abs_diff)[0:2]
        if lineApprox:
            closest_vals = arr[closest_inds]
            dist = closest_vals[1] - closest_vals[0]
            weights = (closest_vals - x) / dist
            return [(closest_inds[0], weights[1]), (closest_inds[1], -weights[0])]
        
        return [(closest_inds[0], 1)]

        
    def predict(self, value: int|float, lineApprox: bool = True):
        inds = self._find_ind_(self.state_values, value, lineApprox=lineApprox)
        pred = 0
        for state, w in inds:
            pred += np.argmax(self.Q[state, :]) * w
        return pred



def some_loss(dist, n):
    return (dist - n)**2 + 1.0

if __name__ == '__main__':
    print('main() from Model.py')

    obj = OptimalDist(grid_params=[(0, 11, 0.5), (0, 11, 1)])
    obj.fit(some_loss)
