#GSKNN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

# read data
df = pd.read_excel('../BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)

# Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using the GridSearchCV() function for multi-parameter tuning
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
neigh = KNeighborsRegressor()
grid_search = GridSearchCV(neigh, parameters, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)
print('best parameters：', grid_search.best_params_)

# Build KNN model
model = KNeighborsRegressor(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
print('KNN-R^2：', round(r2_score(y_test, y_pred), 4))
print('KNN-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('KNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('KNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

#model predictions
X_pred = pd.read_excel('../BSU_D_data_T1.xlsx')
y_pred3 = model.predict(X_pred)
y_test1 = pd.read_excel('../BSU_D_data3-2.xlsx')
y_test1=y_test1.iloc[:,0]
#Model evaluation
print('KNN-R^2：', round(r2_score(y_test1, y_pred3), 4))
print('KNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred3), 4)))
print('KNN-EVS:', round(explained_variance_score(y_test1, y_pred3), 4))
print('KNN-MAE:', round(mean_absolute_error(y_test1, y_pred3), 4))

#####################################################################################
#####################################################################################
#RSKNN
# Using the RandomizedSearchCV function for multi-parameter tuning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import time
from sklearn.metrics import accuracy_score

# read data
df = pd.read_excel('../BSU_data2.xlsx')
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# build model
knn = KNeighborsRegressor()
# Define parameter space
param_space = {'n_neighbors': range(1, 20),
               'weights': ['uniform', 'distance']}
# Create a random search object
random_search = RandomizedSearchCV(knn, param_space, n_iter=10, scoring='r2', cv=5)
# Perform parameter optimization
random_search.fit(X_train, y_train)
# Output the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)
# Rebuild the KNN model using optimal parameters
best_knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
best_knn.fit(X_train, y_train)
# make predictions
y_pred = best_knn.predict(X_test)

# Model evaluation
print('KNN-R^2：', round(r2_score(y_test, y_pred), 4))
print('KNN-RMSE:', np.sqrt(round(mean_squared_error(y_test, y_pred), 4)))
print('KNN-EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('KNN-MAE:', round(mean_absolute_error(y_test, y_pred), 4))

#model predictions
X_pred = pd.read_excel('../BSU_D_data_T1.xlsx')
y_pred3 = model.predict(X_pred)
y_test1 = pd.read_excel('../BSU_D_data3-2.xlsx')
y_test1=y_test1.iloc[:,0]
#Model evaluation
print('KNN-R^2：', round(r2_score(y_test1, y_pred3), 4))
print('KNN-RMSE:', np.sqrt(round(mean_squared_error(y_test1, y_pred3), 4)))
print('KNN-EVS:', round(explained_variance_score(y_test1, y_pred3), 4))
print('KNN-MAE:', round(mean_absolute_error(y_test1, y_pred3), 4))

#Save model
import joblib
joblib.dump(model, "..\RSKNN_new.pickle")
model = joblib.load("..\RSKNN_new.pickle")
###########################################################################################
###########################################################################################
