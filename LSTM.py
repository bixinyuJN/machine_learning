#ACOLSTM
from numpy.random import random as rand
import matplotlib.pyplot as plt
import warnings, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import seaborn as sns

warnings.filterwarnings(action='ignore')

def calc_f(X):
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

def calc_e(X):
    ee = 0
    e1 = X[0] + X[1] - 6
    ee += max(0, e1)
    
    e2 = 3 * X[0] - 2 * X[1] - 5
    ee += max(0, e2)
    return ee

def update_best(parent, parent_fitness, parent_e, child, child_fitness, child_e, X_train, X_test, y_train, y_test):

    if abs(parent[0]) > 0:  
        units = int(abs(parent[0])) * 10 + 48  
    else:
        units = int(abs(parent[0])) + 48  

    if abs(parent[1]) > 0:  
        epochs = int(abs(parent[1])) * 10 + 10  
    else:
        epochs = int(abs(parent[1])) + 20  

    lstm = Sequential()  
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    lstm.add(LSTM(units=units))  
    lstm.add(Dense(10, activation='relu'))  
    lstm.add(Dense(1))  
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  
    score = lstm.evaluate(X_test, y_test, batch_size=128)  

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

m = 20  
G_max = 2  
Rho = 0.9  
P0 = 0.2  
XMAX = 2  
XMIN = 1  
YMAX = 0  
YMIN = -1  
step = 0.1  
P = np.zeros(shape=(G_max, m))  
fitneess_value_list = []  

def initialization():

    X = np.zeros(shape=(m, 2))  
    Tau = np.zeros(shape=(m,))  
    for i in range(m):  
        X[i, 0] = np.random.uniform(XMIN, XMAX, 1)[0]  
        X[i, 1] = np.random.uniform(YMIN, YMAX, 1)[0]  
        Tau[i] = calc_f(X[i])  
    return X, Tau

def position_update(NC, P, X, X_train, X_test, y_train, y_test):
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

def Update_information(Tau, X):

    for i in range(m):  
        Tau[i] = (1 - Rho) * Tau[i] + calc_f(X[i])  
    return Tau

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
    
    best_units, best_epochs = aco(X_train, X_test, y_train, y_test)

    if abs(best_units) > 0:  
        best_units = int(abs(best_units)) * 10 + 48  
    else:
        best_units = int(abs(best_units)) + 48  

    if abs(best_epochs) > 0:  
        best_epochs = int(abs(best_epochs)) * 10 + 100  
    else:
        best_epochs = (int(abs(best_epochs)) + 200)  

    lstm = Sequential()  
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    lstm.add(LSTM(units=best_units))  
    lstm.add(Dense(10, activation='relu'))  
    lstm.add(Dense(1))  
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=64)  
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
    y_test1=y_test1.iloc[:,0]

    print('LSTM-R^2：', round(r2_score(y_test1, y_pred3), 4))
    import numpy as np
    print('LSTM-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred3), 4)))
    print('LSTM-EVS:', round(explained_variance_score(y_test1, y_pred3), 4))
    print('LSTM-MAE:', round(mean_absolute_error(y_test1, y_pred3), 4))
############################################################################################################################
############################################################################################################################
#BOLSTM
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
    y_test1=y_test1.iloc[:,0]

    print('LSTM-R^2：', round(r2_score(y_test1, y_pred3), 4))
    import numpy as np
    print('LSTM-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred3), 4)))
    print('LSTM-EVS:', round(explained_variance_score(y_test1, y_pred3), 4))
    print('LSTM-MAE:', round(mean_absolute_error(y_test1, y_pred3), 4))
############################################################################################################################
############################################################################################################################
#HHOLSTM
import numpy as np  
from numpy.random import rand  
import math  
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
import pandas as pd  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import plot_model  
import keras.layers as layers  
import keras.backend as K  
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  
import seaborn as sns  

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')  
    for i in range(N):  
        for d in range(dim):  
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()  

    return X  

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')  
    for i in range(N):  
        for d in range(dim):  
            if X[i, d] > thres:  
                Xbin[i, d] = 1  
            else:
                Xbin[i, d] = 0  

    return Xbin  

def boundary(x, lb, ub):
    if x < lb:  
        x = lb  
    if x > ub:  
        x = ub  
    return x  

def levy_distribution(beta, dim):
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)  
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  
    sigma = (nume / deno) ** (1 / beta)  
    u = np.random.randn(dim) * sigma  
    v = np.random.randn(dim)  
    step = u / abs(v) ** (1 / beta)  
    LF = 0.01 * step  
    return LF  

def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:  
        units = int(abs(x[0])) * 10  
    else:
        units = int(abs(x[0])) + 16  

    if abs(x[1]) > 0:  
        epochs = int(abs(x[1])) * 10  
    else:
        epochs = int(abs(x[1])) + 10  

    lstm = Sequential()  
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    lstm.add(LSTM(units=units))  
    lstm.add(Dense(10, activation='relu'))  
    lstm.add(Dense(1))  
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  
    score = lstm.evaluate(X_test, y_test, batch_size=128)  

    fitness_value = (1 - float(score[1]))  
    return fitness_value  

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
    
    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)
    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)  

    N = 10  
    T = 2  
    opts = {'N': N, 'T': T}

    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:  
        best_units = int(abs(fmdl[0][0])) * 10 + 48  
    else:
        best_units = int(abs(fmdl[0][0])) + 48  

    if abs(fmdl[0][1]) > 0:  
        best_epochs = int(abs(fmdl[0][1])) * 10 + 60  
    else:
        best_epochs = (int(abs(fmdl[0][1])) + 100)  

    lstm = Sequential()  
    lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    lstm.add(LSTM(units=best_units))  
    lstm.add(Dense(10, activation='relu'))  
    lstm.add(Dense(1))  
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=64)
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
############################################################################################################################
############################################################################################################################
#GSLSTM
import numpy as np  
from numpy.random import rand  
import math  
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
import pandas as pd  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import plot_model  
import keras.layers as layers  
import keras.backend as K  
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  
import seaborn as sns  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import warnings, pandas as pd, numpy as np  
warnings.filterwarnings(action='ignore')  

import time
start_time = time.time()  
df = pd.read_excel('..\BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def create_lstm_model(units=128, epochs=10):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    model.add(LSTM(units=units))  
    model.add(Dense(10, activation='relu'))  
    model.add(Dense(1))  
    model.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  
    return model

lstm_model = KerasClassifier(build_fn=create_lstm_model)
param_grid = {'units': [20,30,40],
              'epochs': [30, 40,50]}
grid_search = GridSearchCV(lstm_model, param_grid, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_lstm_model = create_lstm_model(units=best_params['units'], epochs=best_params['epochs'])
y_pred = best_lstm_model.predict(X_test, batch_size=10)
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
############################################################################################################################
############################################################################################################################
#RSLSTM
import numpy as np  
from numpy.random import rand  
import math  
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  
import pandas as pd  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import plot_model  
import keras.layers as layers  
import keras.backend as K  
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  
import seaborn as sns  
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import warnings, pandas as pd, numpy as np  
warnings.filterwarnings(action='ignore')  

df = pd.read_excel('..\BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def create_lstm_model(units=128, epochs=10):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    model.add(LSTM(units=units))  
    model.add(Dense(10, activation='relu'))  
    model.add(Dense(1))  
    model.compile(loss='mean_squared_error',
                 optimizer='adam',
                 metrics=['mse'])  
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  
    return model

lstm_model = KerasClassifier(build_fn=create_lstm_model)
param_dist = {'units': [20,30,40,50],
              'epochs': [20,30,40,50]}
random_search = RandomizedSearchCV(lstm_model, param_dist, n_iter=3, scoring='accuracy')
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Best Parameters:", best_params)
best_lstm_model = create_lstm_model(units=best_params['units'], epochs=best_params['epochs'])
y_pred = best_lstm_model.predict(X_test, batch_size=10)
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