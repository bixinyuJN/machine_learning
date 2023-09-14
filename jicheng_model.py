#ACOCNN
from numpy.random import random as rand
import matplotlib.pyplot as plt
import warnings, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
import seaborn as sns

warnings.filterwarnings(action='ignore')

# =======Define objective function=====
def calc_f(X):
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

# ====Define the penalty function======
def calc_e(X):
    ee = 0
    e1 = X[0] + X[1] - 6
    ee += max(0, e1)
    e2 = 3 * X[0] - 2 * X[1] - 5
    ee += max(0, e2)
    return ee

# ===Define the selection operation function between children and parents====
def update_best(parent, parent_fitness, parent_e, child, child_fitness, child_e, X_train, X_test, y_train, y_test):
    """
             For different problems, the threshold of the penalty term should be appropriately selected. In this example the threshold is 0.1
             :param parent: parent individual
             :param parent_fitness: parent fitness value
             :param parent_e: parent penalty item
             :param child: child individual
             :param child_fitness child fitness value
             :param child_e: child penalty term
             :return: The better of the parent and offspring, fitness, penalty item
    """

    if abs(parent[0]) > 0:
        units = int(abs(parent[0])) * 10
    else:
        units = int(abs(parent[0])) + 16

    if abs(parent[1]) > 0:
        epochs = int(abs(parent[1])) * 10
    else:
        epochs = int(abs(parent[1])) + 10

    # Establish and train a convolutional neural network regression model
    cnn_model = Sequential()
    cnn_model.add(
            Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu',
                   padding='valid'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=units, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                          optimizer='adam',
                          metrics=['mse'])
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    score = cnn_model.evaluate(X_test, y_test, batch_size=128)

    fitness_value = (1 - float(score[1]))

    parent_e = parent_e + fitness_value
    child_e = child_e + fitness_value

    if parent_e <= 0.1 and child_e <= 0.1:
        if parent_fitness <= child_fitness:
            return parent, parent_fitness, parent_e
        else:
            return child, child_fitness, child_e
    if parent_e < 0.1 and child_e >= 0.1:
        return parent, parent_fitness, parent_e
    if parent_e >= 0.1 and child_e < 0.1:
        return child, child_fitness, child_e
    if parent_fitness <= child_fitness:
        return parent, parent_fitness, parent_e
    else:
        return child, child_fitness, child_e

# =======================Initialization parameters==========================
m = 20 # Number of ants
G_max = 10 # Maximum number of iterations
Rho = 0.9 # Pheromone evaporation coefficient
P0 = 0.2 # Transition probability constant
XMAX = 2 # Search for the maximum value of variable x
XMIN = 1 # Search for the minimum value of variable x
YMAX = 0 # Search for the maximum value of variable y
YMIN = -1 # Search for the minimum value of variable y
step = 0.1 # Local search step size
P = np.zeros(shape=(G_max, m)) #State transition matrix
fitneess_value_list = [] # Iteratively record the optimal objective function value

# =======================Define the initialized ant colony position and pheromone function==========================
def initialization():
    """
    :return: Initializing the ant colony and initial pheromones
    """
    X = np.zeros(shape=(m, 2))
    Tau = np.zeros(shape=(m,))
    for i in range(m):
        X[i, 0] = np.random.uniform(XMIN, XMAX, 1)[0]
        X[i, 1] = np.random.uniform(YMIN, YMAX, 1)[0]
        Tau[i] = calc_f(X[i])
    return X, Tau

# ===Define location update function====
def position_update(NC, P, X, X_train, X_test, y_train, y_test):
    """
         :param NC: current iteration number
         :param P: state transition matrix
         :param X: ant colony
         :return: Ant ColonyX
    """
    lamda = 1 / (NC + 1)
    for i in range(m):
        if P[NC, i] < P0:
            temp1 = X[i, 0] + (2 * np.random.random() - 1) * step * lamda
            temp2 = X[i, 1] + (2 * np.random.random() - 1) * step * lamda
        else:
            temp1 = X[i, 0] + (XMAX - XMIN) * (np.random.random() - 0.5)
            temp2 = X[i, 0] + (YMAX - YMIN) * (np.random.random() - 0.5)

        if (temp1 < XMIN) or (temp1 > XMAX):
            temp1 = np.random.uniform(XMIN, XMAX, 1)[0]
        if (temp2 < YMIN) or (temp2 > YMAX):
            temp2 = np.random.uniform(YMIN, YMAX, 1)[0]

        children = np.array([temp1, temp2])
        children_fit = calc_f(children)
        children_e = calc_e(children)
        parent = X[i]
        parent_fit = calc_f(parent)
        parent_e = calc_e(parent)
        pbesti, pbest_fitness, pbest_e = update_best(parent, parent_fit, parent_e, children, children_fit, children_e,
                                                     X_train, X_test, y_train, y_test)
        X[i] = pbesti
    return X

# ======Pheromone Update============
def Update_information(Tau, X):
    """
         :param Tau: pheromone
         :param X: ant colony
         :return: Tau pheromone
         """
    for i in range(m):
        Tau[i] = (1 - Rho) * Tau[i] + calc_f(X[i])
    return Tau

# =============Define the main function of the ant colony optimization algorithm======================
def aco(X_train, X_test, y_train, y_test):
    X, Tau = initialization()
    for NC in tqdm(range(G_max)):
        BestIndex = np.argmin(Tau)
        Tau_best = Tau[BestIndex]
        for i in range(m):
            P[NC, i] = np.abs((Tau_best - Tau[i])) / np.abs(Tau_best) + 0.01
        X = position_update(NC, P, X, X_train, X_test, y_train, y_test)
        Tau = Update_information(Tau, X)

        index = np.argmin(Tau)
        value = Tau[index]
        fitneess_value_list.append(calc_f(X[index]))

    min_index = np.argmin(Tau)
    minX = X[min_index, 0]
    minY = X[min_index, 1]
    minValue = calc_f(X[min_index])

    return minX, minY

if __name__ == '__main__':
    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df.y
    X = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)
    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)

    # Call the ant colony optimization algorithm
    best_units, best_epochs = aco(X_train, X_test, y_train, y_test)

    if abs(best_units) > 0:
        best_units = int(abs(best_units)) * 10 + 48
    else:
        best_units = int(abs(best_units)) + 48

    if abs(best_epochs) > 0:
        best_epochs = int(abs(best_epochs)) * 10 + 60
    else:
        best_epochs = (int(abs(best_epochs)) + 100)

    # Apply the optimized optimal parameter values to build a convolutional neural network regression model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1),
                         activation='relu'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=best_units, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])

    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=64)
    y_pred = cnn_model.predict(X_test, batch_size=10)

    print('CNN-R^2：', round(r2_score(y_test, y_pred), 4))
    import numpy as np
    print('CNN-MAE:',np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    #prediction
    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
    y_pred1 = cnn_model.predict(X_pred, batch_size=10)
    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1=y_test1.iloc[:,0]
    print('CNN-R^2：', round(r2_score(y_test1, y_pred1), 4))
    import numpy as np
    print('CNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
################################################################################################################
################################################################################################################
# BOLSTM
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import keras.layers as layers
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model

warnings.filterwarnings(action='ignore')


def bayesopt_objective_lstm(units, epochs):
    lstm = Sequential()
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm.add(LSTM(units=int(units)))
    lstm.add(Dense(10, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam', metrics='mse')
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epochs), batch_size=64)
    score = lstm.evaluate(X_test, y_test, batch_size=128)

    return score[1]


def param_bayes_opt_lstm(init_points, n_iter):
    opt = BayesianOptimization(bayesopt_objective_lstm
                               , param_grid_simple
                               , random_state=7)

    opt.maximize(init_points=init_points
                 , n_iter=n_iter
                 )

    params_best = opt.max['params']
    score_best = opt.max['target']

    return params_best, score_best


def bayes_opt_validation_lstm(params_best):
    lstm = Sequential()
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm.add(LSTM(units=int(params_best['units'])))
    lstm.add(Dense(10, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']), batch_size=64)

    score = lstm.evaluate(X_test, y_test, batch_size=128)

    return score[1]


if __name__ == '__main__':
    df = pd.read_excel('..\BSU_data2.xlsx')

    y = df['y']
    X = df.drop('y', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)

    param_grid_simple = {'units': (50.0, 100.0)
        , 'epochs': (50.0, 100.0)
                         }
    params_best, score_best = param_bayes_opt_lstm(10, 10)

    validation_score = bayes_opt_validation_lstm(params_best)

    lstm = Sequential()
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm.add(LSTM(int(params_best['units'])))
    lstm.add(Dense(10, activation='relu'))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])
    history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']),
                       batch_size=64)
    y_pred = lstm.predict(X_test, batch_size=10)

    print('LSTM-R^2：', round(r2_score(y_test, y_pred), 4))
    import numpy as np

    print('LSTM-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('LSTM-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('LSTM-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
    y_pred3 = lstm.predict(X_pred, batch_size=10)
    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1 = y_test1.iloc[:, 0]

    print('LSTM-R^2：', round(r2_score(y_test1, y_pred3), 4))
    import numpy as np

    print('LSTM-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred3), 4)))
    print('LSTM-EVS:', round(explained_variance_score(y_test1, y_pred3), 4))
    print('LSTM-MAE:', round(mean_absolute_error(y_test1, y_pred3), 4))

##################################################################################################################
##################################################################################################################
#Model integration solution 1,ensemble model(ACOCNN+BOLSTM)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, LSTM, concatenate
# Define ensemble regression model
def create_regression_model(cnn_model, lstm):
    cnn_output_shape = cnn_model.output_shape[1:]
    lstm_input_shape = lstm.input_shape[1:]
    cnn_output = cnn_model.output
    cnn_output = tf.reshape(cnn_output, (-1, np.prod(cnn_output_shape)))
    lstm_input = lstm.input
    lstm_output = lstm.output
    merged = concatenate([cnn_output, lstm_output])
    output = Dense(1)(merged)
    model = Model(inputs=[cnn_model.input, lstm_input], outputs=output)
    return model
regression_model = create_regression_model(cnn_model, lstm)
regression_model.compile(optimizer='adam', loss='mse')
regression_model.summary()
y_pred = regression_model.predict([X_pred, X_pred2])
y_test = pd.read_excel('..\BSU_D_data3-2.xlsx')
y_test = y_test.iloc[:, 0]
print('CNN-R^2：', round(r2_score(y_test, y_pred), 4))
import numpy as np
print('CNN-MAE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))
history = regression_model.fit(X_train, X_train1, y_train, y_train1, validation_data=(X_test, X_test1, y_test, y_test1))

##################################################################################################################
#Model integration solution 2, integrate labels into the BOSVR model, and compare with the cnn and lstm integrated models

from numpy.random import random as rand
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score

# =======Define objective function=====
def calc_f(X):
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

# ====Define the penalty function======
def calc_e(X):
    ee = 0
    e1 = X[0] + X[1] - 6
    ee += max(0, e1)
    e2 = 3 * X[0] - 2 * X[1] - 5
    ee += max(0, e2)
    return ee

# ===Define the selection operation function between children and parents====
def update_best(parent, parent_fitness, parent_e, child, child_fitness, child_e, X_train, X_test, y_train, y_test):
    """
             For different problems, the threshold of the penalty term should be appropriately selected. In this example the threshold is 0.1
             :param parent: parent individual
             :param parent_fitness: parent fitness value
             :param parent_e: parent penalty item
             :param child: child individual
             :param child_fitness child fitness value
             :param child_e: child penalty term
             :return: The better of the parent and offspring, fitness, penalty item
             """

    svr = SVR(kernel='linear', C=abs(parent[0]), gamma=abs(parent[1]) * 10).fit(X_train, y_train)
    cv_accuracies = cross_val_score(svr, X_test, y_test, cv=3,
                                    scoring='r2')

    accuracies = cv_accuracies.mean()
    fitness_value = 1 - accuracies
    parent_e = parent_e + fitness_value
    child_e = child_e + fitness_value
    if parent_e <= 0.1 and child_e <= 0.1:
        if parent_fitness <= child_fitness:
            return parent, parent_fitness, parent_e
        else:
            return child, child_fitness, child_e
    if parent_e < 0.1 and child_e >= 0.1:
        return parent, parent_fitness, parent_e
    if parent_e >= 0.1 and child_e < 0.1:
        return child, child_fitness, child_e
    if parent_fitness <= child_fitness:
        return parent, parent_fitness, parent_e
    else:
        return child, child_fitness, child_e

# =======================Initialization parameters==========================
m = 20 # Number of ants
G_max = 10 # Maximum number of iterations
Rho = 0.9 # Pheromone evaporation coefficient
P0 = 0.2 # Transition probability constant
XMAX = 2 # Search for the maximum value of variable x
XMIN = 1 # Search for the minimum value of variable x
YMAX = 0 # Search for the maximum value of variable y
YMIN = -1 # Search for the minimum value of variable y
step = 0.1 # Local search step size
P = np.zeros(shape=(G_max, m)) #State transition matrix
fitneess_value_list = [] # Iteratively record the optimal objective function value

# =======================Define the initialized ant colony position and pheromone function==========================
def initialization():
    """
    :return: Initializing the ant colony and initial pheromones
    """
    X = np.zeros(shape=(m, 2))
    Tau = np.zeros(shape=(m,))
    for i in range(m):
        X[i, 0] = np.random.uniform(XMIN, XMAX, 1)[0]
        X[i, 1] = np.random.uniform(YMIN, YMAX, 1)[0]
        Tau[i] = calc_f(X[i])
    return X, Tau

# ===Define location update function====
def position_update(NC, P, X, X_train, X_test, y_train, y_test):
    """
         :param NC: current iteration number
         :param P: state transition matrix
         :param X: ant colony
         :return: Ant ColonyX
    """
    lamda = 1 / (NC + 1)
    # =======location update==========
    for i in range(m):
        # ===local search===
        if P[NC, i] < P0:
            temp1 = X[i, 0] + (2 * np.random.random() - 1) * step * lamda
            temp2 = X[i, 1] + (2 * np.random.random() - 1) * step * lamda
        # ===global search===
        else:
            temp1 = X[i, 0] + (XMAX - XMIN) * (np.random.random() - 0.5)
            temp2 = X[i, 0] + (YMAX - YMIN) * (np.random.random() - 0.5)

        # =====Boundary processing=====
        if (temp1 < XMIN) or (temp1 > XMAX):
            temp1 = np.random.uniform(XMIN, XMAX, 1)[0]
        if (temp2 < YMIN) or (temp2 > YMAX):  # 判断
            temp2 = np.random.uniform(YMIN, YMAX, 1)[0]  #

        # =====Determine whether the ants are moving (choose the better one)=====
        children = np.array([temp1, temp2])
        children_fit = calc_f(children)
        children_e = calc_e(children)
        parent = X[i]
        parent_fit = calc_f(parent)
        parent_e = calc_e(parent)
        pbesti, pbest_fitness, pbest_e = update_best(parent, parent_fit, parent_e, children, children_fit, children_e,
                                                     X_train, X_test, y_train, y_test)
        X[i] = pbesti
    return X

# ======Pheromone Update============
def Update_information(Tau, X):
    """
         :param Tau: pheromone
         :param X: ant colony
         :return: Tau pheromone
    """
    for i in range(m):
        Tau[i] = (1 - Rho) * Tau[i] + calc_f(X[i])
    return Tau

# =============Define the main function of the ant colony optimization algorithm======================
def aco(X_train, X_test, y_train, y_test):
    X, Tau = initialization()
    for NC in tqdm(range(G_max)):
        BestIndex = np.argmin(Tau)
        Tau_best = Tau[BestIndex]

        for i in range(m):
            P[NC, i] = np.abs((Tau_best - Tau[i])) / np.abs(Tau_best) + 0.01
        # =======location update==========
        X = position_update(NC, P, X, X_train, X_test, y_train, y_test)

        # =====Update pheromones========
        Tau = Update_information(Tau, X)

        # =====Record the optimal objective function value========
        index = np.argmin(Tau)
        value = Tau[index]
        fitneess_value_list.append(calc_f(X[index]))

    # =====Print results=======
    min_index = np.argmin(Tau)
    minX = X[min_index, 0]
    minY = X[min_index, 1]
    minValue = calc_f(X[min_index])

    best_C = minX
    best_gamma = minY

    # =====Visualization=======
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(fitneess_value_list, label='iteration curve')
    plt.legend()
    plt.show()

    return best_C, best_gamma

if __name__ == '__main__':
    # Read data
    df = pd.read_excel('..\BSU_data2.xlsx')
    import time
    start_time = time.time()
    y = df.y
    X = df.drop('y', axis=1)

    # Divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the ant colony optimization algorithm
    best_C, best_gamma = aco(X_train, X_test, y_train, y_test)

    # Apply the optimized optimal parameter values to build a support vector machine regression model
    svr = SVR(kernel='linear',C=abs(best_C), gamma=abs(best_gamma) * 10)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    print('SVR-R^2：', round(r2_score(y_test, y_pred), 4))
    import numpy as np
    print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('SVR-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('SVR-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    #prediction
    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    y_pred1 = svr.predict(X_pred)  # 预测
    y_pred1 = np.transpose(y_pred1)

    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1=y_test1.iloc[:,0]
    print('SVR-R^2：', round(r2_score(y_test1, y_pred1), 4))
    import numpy as np
    print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
    print('SVR-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
    print('SVR-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))

