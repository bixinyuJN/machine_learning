#1.HistGradientBoostingRegressor
import numpy as np
import pandas as pd  
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_excel('../BsuMAC_d.xlsx')
y = df['y'].values
X = df.drop('y', axis=1).values
model = HistGradientBoostingRegressor()
missing_indexes = np.isnan(X).any(axis=1)
for i in range(X.shape[1]):
    feature = X[:, i]
    X_train = feature[~missing_indexes].reshape(-1, 1)
    y_train = y[~missing_indexes]
    X_test = feature[missing_indexes].reshape(-1, 1)
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    X[missing_indexes, i] = predicted_values

########################################################################################
#2.DBSCAN
import numpy as np
from sklearn.cluster import DBSCAN
df = pd.read_excel('../BSU_data2.xlsx')
y = df['y'].values
X = df.drop('y', axis=1).values
eps = 0.5  
min_samples = 2  
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)
outliers = np.where(labels == -1)[0]
print("Detected outliers:", outliers)

####################################################################################
#3.PCA
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_excel('../BSU_data2.xlsx')
y = df['y'].values
X = df.drop('y', axis=1).values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca)
##############################################################################################
#4.Data normalization
import numpy as np
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('..\TRN_d.xlsx')
X = df.values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
Res = X_normalized.tolist()
def writedatatoexcal(data):
    wb = Workbook()
    sheet = wb.active
    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.cell(i + 1, j + 1).value = data[i][j]
    wb.save('..\TRN_d.xlsx')
    wb.close()
from openpyxl import Workbook
writedatatoexcal(Res)
########################################################################################################
#5.Feature Selection---Particle Swarm Optimization Dataset
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


num_particles = 20
max_iterations = 100
c1 = 2.0  
c2 = 2.0  
w = 0.7   

df = pd.read_excel('../BSU_data2.xlsx')
y = df['y'].values
X = df.drop('y', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X_train.shape[1]  
num_selected_features = 2  

particle_position = []
particle_velocity = []
for _ in range(num_particles):
    particle_position.append(np.random.choice([0, 1], size=num_features))
    particle_velocity.append(np.zeros(num_features))

global_best_position = np.zeros(num_features)
global_best_fitness = float('-inf')

for iteration in range(max_iterations):
    for particle_index in range(num_particles):
        
        selected_features = np.where(particle_position[particle_index] == 1)[0]

        if len(selected_features) == 0:
            continue  

        classifier = KNeighborsRegressor(n_neighbors=3)
        classifier.fit(X_train[:, selected_features], y_train)
        fitness = classifier.score(X_test[:, selected_features], y_test)

        if fitness > global_best_fitness:
            global_best_position = particle_position[particle_index]
            global_best_fitness = fitness

        particle_velocity[particle_index] = (w * particle_velocity[particle_index] +
                                             c1 * random.random() * (particle_position[particle_index] - particle_position[particle_index]) +
                                             c2 * random.random() * (global_best_position - particle_position[particle_index]))
        particle_position[particle_index] = np.where(particle_velocity[particle_index] > 0, 1, 0)

selected_features = np.where(global_best_position == 1)[0]
