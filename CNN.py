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
##############################################################################################################################################
##############################################################################################################################################
#BOCNN
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Dropout
import keras.layers as layers
import keras.backend as K

warnings.filterwarnings(action='ignore')

# Bayesian objective function optimization convolutional neural network classification model
def bayesopt_objective_cnn(filters, units, epochs):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=int(filters), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu', padding='valid'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=int(units), activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                     optimizer='adam',
                     metrics=['mse'])
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(epochs), batch_size=64)
    score = cnn_model.evaluate(X_test, y_test, batch_size=128)

    return score[1]


# Bayesian Optimizer
def param_bayes_opt_cnn(init_points, n_iter):
    opt = BayesianOptimization(bayesopt_objective_cnn
                               , param_grid_simple
                               , random_state=7)

    opt.maximize(init_points=init_points
                 , n_iter=n_iter
                 )
    params_best = opt.max['params']
    score_best = opt.max['target']

    return params_best, score_best


# Customize the verification function and return the optimal parameters of bayes_opt
def bayes_opt_validation_cnn(params_best):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=int(params_best['filters']), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=int(params_best['units'])))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                     optimizer='adam',
                     metrics=['mse'])
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']),
                 batch_size=64)

    score = cnn_model.evaluate(X_test, y_test, batch_size=128)

    return score[1]


if __name__ == '__main__':
    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)

    param_grid_simple = {'filters': (2.0, 5.0),
                         'units': (20.0, 100.0)
        , 'epochs': (50.0, 100.0)
                         }

    params_best, score_best = param_bayes_opt_cnn(10, 10)
    validation_score = bayes_opt_validation_cnn(params_best)

    # Optimal parameter construction model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=int(params_best['filters']), kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu'))  # 1维卷积层
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(int(params_best['units']), activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                     optimizer='adam',
                     metrics=['mse'])
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(params_best['epochs']),
                           batch_size=64)

    y_pred = cnn_model.predict(X_test, batch_size=10)
    print('CNN-R^2：', round(r2_score(y_test, y_pred), 4))
    import numpy as np
    print('CNN-MAE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    # prediction
    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
    y_pred1 = cnn_model.predict(X_pred, batch_size=10)
    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1 = y_test1.iloc[:, 0]
    print('CNN-R^2：', round(r2_score(y_test1, y_pred1), 4))
    import numpy as np
    print('CNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
##############################################################################################################################################
##############################################################################################################################################
#HHOCNN
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import seaborn as sns


# Define initialization position function
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


# Define conversion function
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


# Define boundary processing functions
def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


# Define Levi's flight function--parameter assignment
def levy_distribution(beta, dim):

    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    # Parameter u & v
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)

    step = u / abs(v) ** (1 / beta)
    LF = 0.01 * step

    return LF


# Define error rate calculation function
def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:
        units = int(abs(x[0])) * 10
    else:
        units = int(abs(x[0])) + 16

    if abs(x[1]) > 0:
        epochs = int(abs(x[1])) * 10
    else:
        epochs = int(abs(x[1])) + 10

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

    return fitness_value


# Define objective function
def Fun(X_train, y_train, X_test, y_test, x, opts):
    alpha = 0.99
    beta = 1 - alpha
    max_feat = len(x)
    num_feat = np.sum(x == 1)
    if num_feat == 0:
        cost = 1
    else:
        error = error_rate(X_train, y_train, X_test, y_test, x, opts)
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost


# Define the main function of Harris Eagle optimization algorithm
def jfs(X_train, y_train, X_test, y_test, opts):
    ub = 1
    lb = 0
    thres = 0.5
    beta = 1.5

    N = opts['N']
    max_iter = opts['T']
    if 'beta' in opts:
        beta = opts['beta']

    dim = np.size(X_train, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    X = init_position(lb, ub, N, dim)

    fit = np.zeros([N, 1], dtype='float')
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:

        Xbin = binary_conversion(X, thres, N, dim)

        for i in range(N):
            fit[i, 0] = Fun(X_train, y_train, X_test, y_test, Xbin[i, :], opts)
            if fit[i, 0] < fitR:
                Xrb[0, :] = X[i, :]
                fitR = fit[i, 0]

        curve[0, t] = fitR.copy()

        t += 1

        X_mu = np.zeros([1, dim], dtype='float')
        X_mu[0, :] = np.mean(X, axis=0)

        for i in range(N):
            E0 = -1 + 2 * rand()
            E = 2 * E0 * (1 - (t / max_iter))
            if abs(E) >= 1:
                q = rand()
                if q >= 0.5:
                    k = np.random.randint(low=0, high=N)
                    r1 = rand()
                    r2 = rand()
                    for d in range(dim):
                        X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif q < 0.5:
                    r3 = rand()
                    r4 = rand()
                    for d in range(dim):
                        X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])


            elif abs(E) < 1:
                J = 2 * (1 - rand())
                r = rand()
                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = Xrb[0, d] - E * abs(DX)
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif r < 0.5 and abs(E) >= 0.5:
                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]

                elif r < 0.5 and abs(E) < 0.5:
                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]

    return X


if __name__ == '__main__':
    # Read data
    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)
    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)

    N = 10
    T = 2

    opts = {'N': N, 'T': T}

    # Call the main function of Harris Eagle optimization algorithm
    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:
        best_units = int(abs(fmdl[0][0])) * 10 + 48
    else:
        best_units = int(abs(fmdl[0][0])) + 48

    if abs(fmdl[0][1]) > 0:
        best_epochs = int(abs(fmdl[0][1])) * 10 + 60
    else:
        best_epochs = (int(abs(fmdl[0][1])) + 100)


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
    print('CNN-MAE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    # prediction
    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
    y_pred1 = cnn_model.predict(X_pred, batch_size=10)
    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1 = y_test1.iloc[:, 0]
    print('CNN-R^2：', round(r2_score(y_test1, y_pred1), 4))
    import numpy as np
    print('CNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
    print('CNN-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
    print('CNN-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
##############################################################################################################################################
##############################################################################################################################################
#RSCNN
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import warnings, pandas as pd, numpy as np
warnings.filterwarnings(action='ignore')

df = pd.read_excel('..\BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_cnn_model(units=128, epochs=10):
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1),
                         activation='relu'))
    model.add(MaxPooling1D(strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    # cnn_model.add(Dropout(0.5))  # Dropout层
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])
    history =model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    return model

# Create Keras classifier
cnn_model = KerasClassifier(build_fn=create_cnn_model)
param_dist = {'units': [5,10],
              'epochs': [10,20]}
random_search = RandomizedSearchCV(cnn_model, param_dist, n_iter=3, scoring='accuracy')
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Best Parameters:", best_params)
# Rebuild the CNN model using optimal parameters
best_cnn_model = create_cnn_model(units=best_params['units'], epochs=best_params['epochs'])
y_pred = best_cnn_model.predict(X_test)
print('CNN-R^2：', round(r2_score(y_test, y_pred), 4))
import numpy as np
print('CNN-MAE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

# prediction
X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
y_pred1 = cnn_model.predict(X_pred, batch_size=10)
y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
y_test1 = y_test1.iloc[:, 0]
print('CNN-R^2：', round(r2_score(y_test1, y_pred1), 4))
import numpy as np
print('CNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
print('CNN-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
print('CNN-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
########################################################################################################################################
########################################################################################################################################
#GSCNN
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import warnings, pandas as pd, numpy as np
warnings.filterwarnings(action='ignore')

df = pd.read_excel('..\BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_cnn_model(units=128, epochs=10):
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1),
                         activation='relu'))
    model.add(MaxPooling1D(strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])
    history =model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    return model

cnn_model = KerasClassifier(build_fn=create_cnn_model)
param_dist = {'units': [5,15],
              'epochs': [20,30]}
random_search = GridSearchCV(cnn_model, param_dist, scoring='accuracy')
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Best Parameters:", best_params)
best_cnn_model = create_cnn_model(units=best_params['units'], epochs=best_params['epochs'])
y_pred = best_cnn_model.predict(X_test)
print('CNN-R^2：', round(r2_score(y_test, y_pred), 4))
import numpy as np
print('CNN-MAE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('CNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('CNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

# prediction
X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
X_pred = layers.Lambda(lambda X_pred: K.expand_dims(X_pred, axis=-1))(X_pred)
y_pred1 = cnn_model.predict(X_pred, batch_size=10)
y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
y_test1 = y_test1.iloc[:, 0]
print('CNN-R^2：', round(r2_score(y_test1, y_pred1), 4))
import numpy as np
print('CNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
print('CNN-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
print('CNN-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))










