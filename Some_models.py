# from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf

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
    def __init__(self, grid_params:any = [(50, 101, 25), (0., 5., 1.), (0., 5., 1.), (0., 5., 1.)]):
        self.hasFit = False
        self.hasData = False
        self.target = ['d0', 'd1', 'd2']
        self.feature = ['N']
        self.cars = grid_params[0]
        self.dist = grid_params[1:]

    # def __init__(self, n_estimators:int = 100, max_depth: int = 3):
    #     self.n_estimators = n_estimators
    #     self.max_depth = max_depth
    #     self.hasFit = False
        
    
    def _fit_(self, X:pd.DataFrame, Y:pd.DataFrame, n_estimators:int, max_depth:int):
        self.reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth))
        self.reg.fit(X, Y)
        self.hasFit = True
        

    def fit_data(self, data_name:str, n_estimators:int=100, max_depth:int=3) -> None:
        data = pd.read_csv(data_name)
        X = data[self.feature]
        Y = data[self.target]
        self._fit_(X, Y, n_estimators, max_depth)



    def fit_func(self, func:callable, n_estimators:int=100, max_depth:int=3) -> None:
        """
        Обучает модель
        """
        # assert ((data_name != None) or (func != None)), ('\n' + "соси жопу " * 20) * 50 + '\n'
        X = pd.DataFrame(columns=self.feature)
        Y = pd.DataFrame(columns=self.target)

        for n_cars in tqdm.tqdm(range(*self.cars)):
            max_loss = -1.0
            best_dists = []
            for g0 in np.arange(*self.dist[0]):
                for g1 in np.arange(*self.dist[1]):
                    for g2 in np.arange(*self.dist[2]):
                        loss_res = func(g0, g1, g2, n_cars)
                        if loss_res > max_loss:
                            max_loss = loss_res
                            best_dists = [g0, g1, g2]
            X.loc[len(X)] = n_cars
            Y.loc[len(Y)] = best_dists

        self._fit_(X, Y, n_estimators, max_depth)
        

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


    def fit(self, func: callable,
            alpha: float = 0.1,  # шаг обучения (learning rate)
            gamma: float = 0.9,  # дисконт-фактор
            epsilon: float=0.1,  # эпсилон для epsilon-greedy стратегии
            max_step_in_episode: int = 100,
            num_episodes: int = 1000,  # количество эпох
            showInfo: bool = False  # покзаывает результаты обучения после каждой эпохи
    ):
        self.reword_func = func

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
                reward = self.reword_func(x, y)  # награда -значение функции потерь
                
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
    

class OptimalNN():
    def __init__(self, min_N: int|float, max_N: int|float) -> None:
        self.min_N = min_N
        self.max_N = max_N
        self.throughtput = Throughput()


    @tf.function
    def _train_step_(self, n):
        """
        Обновляет параметры модели
        """
        with tf.GradientTape() as tape:
            predictions = self.model(n)
            print(predictions[0])
            val = tf.concat([predictions[0], n], axis=0)
            print(val)
            val = tf.expand_dims(val, axis=0) 
            # g0, g1, g2 = tf.unstack(predictions, axis=1)
            # val = tf.constant([g0[0][0], g1[0][0], g2[0][0], n[0][0]], shape=(1, 4))
            loss = -self.throughtput.model(val)
            print(loss)
    
            # loss = -(self.func(n, g0, g1, g2))  # Минимизируем отрицание функции  TODO:убрать костыль с numpy()[0]
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def fit(self, func: callable, n_iterations: int = 1000, n_hidden_layers: int = 2) -> None:
        self.func = func
        # Создаем модель нейронной сети
        n_neurons = 64
        activation_func = 'relu'
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(1,)))
        for _ in range(n_hidden_layers):
            self.model.add(tf.keras.layers.Dense(n_neurons, activation=activation_func))
        self.model.add(tf.keras.layers.Dense(3))

        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(2)  # Два выходных нейрона для g1 и g2
        # ])

        # Оптимизатор
        self.optimizer = tf.keras.optimizers.Adam()
            
        # for n in np.random.random_integers(self.min_N, self.max_N, n_iterations):
        #     self._train_step_(n)
        self.throughtput.fit('csv_data/test6.csv',test_size=0.7, n_epochs=2)
        for _ in range(n_iterations):
            n = tf.round(tf.random.uniform((1,), minval=self.min_N, maxval=self.max_N, dtype=tf.float32, seed=13))
            self._train_step_(n)


    def predict(self, n: int) -> list:
        """
        Получаем оптимальные значения g1 и g2
        """
        tf_n = tf.constant([[n]])
        pred = list(map(lambda x: x.numpy()[0], tf.unstack(self.model(tf_n), axis=1)))
        return pred


class SimpleNN():
    def __init__(self):
        self.target = ['d0', 'd1', 'd2']
        self.feature = ['N']

    def fit(self, data_filename:str, n_hidden_layers: int = 2, n_epochs: int = 10) -> None:
        data = pd.read_csv(data_filename)
        X = data[self.feature]
        Y = data[self.target]

        n_neurons = 64
        activation_func = 'relu'
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(1,)))
        for _ in range(n_hidden_layers):
            self.model.add(tf.keras.layers.Dense(n_neurons, activation=activation_func))
        self.model.add(tf.keras.layers.Dense(3))

        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, Y, epochs=n_epochs)

    def predict(self, x:int|float|list|np.ndarray):
        val = tf.constant(x, dtype=tf.float32, shape=(len(x),))
        return self.model.predict(val)


class Throughput():
    def __init__(self):
        self.input = ['d0', 'd1', 'd2', 'N']
        self.output = ['throughput']

    def fit(self, data_filename:str, n_epochs:int = 10, test_size:float = 0.8, seed:int = 13) -> None:
        data = pd.read_csv(data_filename)
        input = data[self.input]
        output = data[self.output]
        in_train, in_test, out_train, out_test = train_test_split(input, output, test_size=test_size, random_state=seed)

        # Один вход для всех четырех значений
        input_full = tf.keras.layers.Input(shape=(4,), name='input_full')
        # Разделение входа на три и один
        input1 = tf.keras.layers.Lambda(lambda x: x[:, :3], output_shape=(3,))(input_full)  # Первые три значения
        input2 = tf.keras.layers.Lambda(lambda x: x[:, 3:], output_shape=(1,))(input_full)  # Последнее значение
        # Скрытые слои для первого входа
        hidden1 = tf.keras.layers.Dense(30, activation='relu')(input1)
        hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
        # Объединение второго входа с результатом обработки первого входа
        merged = tf.keras.layers.concatenate([hidden2, input2])
        # Общие скрытые слои
        shared_hidden1 = tf.keras.layers.Dense(30, activation='relu')(merged)
        shared_hidden2 = tf.keras.layers.Dense(30, activation='relu')(shared_hidden1)
        # Выходной слой
        output = tf.keras.layers.Dense(1)(shared_hidden2)
        # Создание модели
        self.model = tf.keras.models.Model(inputs=input_full, outputs=output)
        # Компиляция модели
        self.model.compile(optimizer='adam', loss='mse')

        # Обучение модели
        self.model.fit(in_train, out_train, epochs=n_epochs)

        mse_train = self.model.evaluate(in_train, out_train)
        mse_test = self.model.evaluate(in_test, out_test)
        print("MSE на обучающих данных:", mse_train)
        print("MSE на тестовых данных:", mse_test)

    def predict(self, x:list) -> float:
        val = tf.constant(x, dtype=tf.float32, shape=(1,4))
        return self.model.predict(val)[0]




def some_loss(dist, n):
    return -(dist - n)**2 + 1.0

if __name__ == '__main__':
    print('main() from Model.py')

    obj = OptimalDist(grid_params=[(0, 11, 0.5), (0, 11, 1)])
    obj.fit(some_loss)
