#ACOSVR
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
###################################################################################################################################################
###################################################################################################################################################
#BOSVR
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import sklearn.metrics as metrics
from sklearn.svm import SVR

warnings.filterwarnings(action='ignore')

# Bayesian objective function optimization support vector machine regression model
def bayesopt_objective_svr(C, gamma):
    svr_model = SVR(kernel='linear', C=C, gamma=gamma)
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    validation_acc = cross_validate(svr_model
                                    , X_train, y_train
                                    , scoring='r2'
                                    , cv=cv
                                    , error_score='raise'
                                    )

    return np.mean(validation_acc['test_score'])


# Define Bayesian optimizer function
def param_bayes_opt_svr(init_points, n_iter):
    opt = BayesianOptimization(bayesopt_objective_svr
                               , param_grid_simple
                               , random_state=7)

    # Use optimizer
    opt.maximize(init_points=init_points
                 , n_iter=n_iter  #
                 )

    # Return optimization results
    params_best = opt.max['params']
    score_best = opt.max['target']

    return params_best, score_best


# Customize the verification function to return the optimal parameters of bayes_opt
def bayes_opt_validation_svr(params_best):
    svr_model = SVR(kernel='linear', C=params_best['C'], gamma=params_best['gamma'])
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    validation_acc = cross_validate(svr_model
                                    , X_test, y_test
                                    , scoring='r2'
                                    , cv=cv
                                    )

    return np.mean(validation_acc['test_score'])


if __name__ == '__main__':

    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid_simple = {'C': (0.1, 10.0),
                         'gamma': (0.01, 1.0)
                         }

    params_best, score_best = param_bayes_opt_svr(10, 10)
    validation_score = bayes_opt_validation_svr(params_best)

    bys_svr = SVR(kernel='linear', C=params_best['C'], gamma=params_best['gamma'])
    bys_svr.fit(X_train, y_train)
    y_pred = bys_svr.predict(X_test)

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

########################################################################################################################################
########################################################################################################################################
#HHOSVR
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score

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

# Define Levy flight function
def levy_distribution(beta, dim):
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / abs(v) ** (1 / beta)
    LF = 0.01 * step
    return LF


# Define error rate calculation function
def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:
        gamma = abs(x[0]) / 10
    else:
        gamma = 'scale'

    if abs(x[1]) > 0:
        C = abs(x[1]) / 10
    else:
        C = 1.0

    svr = SVR(kernel='linear', C=C, gamma=gamma).fit(X_train, y_train)
    cv_accuracies = cross_val_score(svr, X_test, y_test, cv=3,
                                    scoring='r2')
    accuracies = cv_accuracies.mean()
    fitness_value = 1 - accuracies
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
    df = pd.read_excel('..\BSU_data2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    N = 10
    T = 10
    opts = {'N': N, 'T': T}

    # Call the main function of Harris Eagle optimization algorithm
    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:
        best_C = abs(fmdl[0][0])
    else:
        best_C = abs(fmdl[0][0]) + 1

    if abs(fmdl[0][1]) > 0:
        best_gamma = abs(fmdl[0][1])
    else:
        best_gamma = (abs(fmdl[0][1]) + 0.1)


    # Apply the optimized optimal parameter values to build a support vector machine regression model
    svr = SVR(kernel='linear', C=best_C, gamma=best_gamma)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    print('SVR-R^2：', round(r2_score(y_test, y_pred), 4))
    import numpy as np
    print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
    print('SVR-EVS:', round(explained_variance_score(y_test, y_pred), 4))
    print('SVR-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

    # prediction
    X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
    y_pred1 = svr.predict(X_pred)  # 预测
    y_pred1 = np.transpose(y_pred1)

    y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
    y_test1 = y_test1.iloc[:, 0]
    print('SVR-R^2：', round(r2_score(y_test1, y_pred1), 4))
    import numpy as np
    print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
    print('SVR-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
    print('SVR-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
#############################################################################################################################################
#############################################################################################################################################
#RSSVR
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import time
import pandas as pd
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score

df = pd.read_excel('../BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr = SVR()
param_space = {'C': np.logspace(-3, 3, 7),
               'gamma': np.logspace(-3, 3, 7)}
random_search = RandomizedSearchCV(svr, param_space, n_iter=10, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X, y)
best_params = random_search.best_params_
print("Best Parameters:", best_params)
best_svr = SVR(**best_params)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print('SVR-R^2：', round(r2_score(y_test, y_pred), 4))
import numpy as np

print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('SVR-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('SVR-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

# prediction
X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
y_pred1 = svr.predict(X_pred)  # 预测
y_pred1 = np.transpose(y_pred1)

y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
y_test1 = y_test1.iloc[:, 0]
print('SVR-R^2：', round(r2_score(y_test1, y_pred1), 4))
import numpy as np

print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
print('SVR-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
print('SVR-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))
#############################################################################################################################################
#############################################################################################################################################
#GSSVR
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

df = pd.read_excel('../BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr = SVR()

param_grid = {'C': np.logspace(-1, 1, 10),
              'gamma': np.logspace(-1, 1, 10)}

grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error')

grid_search.fit(X, y)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_svr = SVR(**best_params)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print('SVR-R^2：', round(r2_score(y_test, y_pred), 4))
import numpy as np

print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('SVR-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('SVR-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

# prediction
X_pred = pd.read_excel('..\BSU_D_data_T1.xlsx')
y_pred1 = svr.predict(X_pred)  # 预测
y_pred1 = np.transpose(y_pred1)

y_test1 = pd.read_excel('..\BSU_D_data3-2.xlsx')
y_test1 = y_test1.iloc[:, 0]
print('SVR-R^2：', round(r2_score(y_test1, y_pred1), 4))
import numpy as np

print('SVR-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred1), 4)))
print('SVR-EVS:', round(explained_variance_score(y_test1, y_pred1), 4))
print('SVR-MAE:', round(mean_absolute_error(y_test1, y_pred1), 4))