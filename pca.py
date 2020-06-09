# PCA analysis of 2d square lattice ising configurations generated from MC

import json
import numpy as np
import matplotlib.pyplot as plt

# load configurations and temperatures
with open('monte_carlo_dataset.json', 'r') as data_file:
	[configurations, temperatures] = json.load(data_file)

# calculate the scatter matrix
X = np.array(configurations)
mean_X = np.mean(X, axis=0)
norm_X = X - mean_X
S_matrix = np.dot(norm_X.T, norm_X)

# eign-system of the scatter matrix
values, vectors = np.linalg.eig(S_matrix)
pairs = [[np.abs(values[i]), vectors[:, i]] for i in range(len(values))]

# sort according to the magnitude of eign-values
def takeFirst(x):
	return x[0]

pairs.sort(key=takeFirst, reverse=True)

# construct the projector P, e.g. to 2D here
P = np.array([pairs[0][1], pairs[1][1]])

# normalized data after projecting to 2 dimension
points = np.transpose(np.real(np.dot(norm_X, P.T)))

# plot the variance ratios
# ratios = np.abs(values)
# plt.scatter(range(1, len(ratios)+1)[0:10], (ratios/np.sum(ratios))[0:10], c='r', edgecolors='#07031a')
# plt.title('The First 10 Relative Ratios of PCA 20*20')
# plt.ylabel('relative ratios')
# plt.yscale('log')
# plt.ylim(0.001, 1)
# plt.show()

# plot the PCA projection fig
plt.figure(figsize=(30, 6))
plt.scatter(points[0], points[1], c=temperatures, cmap='bwr', edgecolors='#07031a', s=20)
plt.title('PCA Result For Ising Monte Carlo 20*20')
plt.xlabel('component one')
plt.ylabel('component two')
plt.colorbar()
plt.show()