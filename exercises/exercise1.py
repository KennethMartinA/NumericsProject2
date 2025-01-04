# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:34:47 2024

@author: kam224
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns
#exercise for numerics
def force_vector(N = int(1e5)):
    force_vec = np.zeros(numElements)
    sample = np.random.uniform(size = N)
    for i in range(1, numElements + 1):

        node = nodes[i]
        force_vec[i - 1] = np.mean(shape_function(sample, node) * (1 - sample)**2)
    return force_vec

def shape_function(x, center_node):
    #return x for the ith shape function
    x = x.copy()
    mask1 = (x >= center_node - dx) & (x < center_node)
    mask2 = (x >= center_node) & (x <= center_node + dx)
    mask3 = (x >= center_node + dx) | (x<= center_node - dx)
    x[mask1] = x[mask1] - center_node + dx
    x[mask2] = dx + center_node - x[mask2]
    x[mask3] = 0
    x[x <= 0] = 0
    x[x > 1] = 0
    return x/dx
       
numElements = 3

dx = 1./numElements
psi = np.zeros(numElements + 1)
nodes = np.linspace(0, 1, num = numElements + 1)
mass = np.matrix("2 , -1, 0; -1, 2, -1; 0, -1, 1") / dx

ones = np.ones(numElements)
mass = np.zeros((numElements, numElements)) + 2 * np.diag(ones) - \
        np.diag(ones[1:], 1) - np.diag(ones[1:], -1)
mass[-1,-1] = 1
mass = mass / dx
force = force_vector()

psi[1:] = np.linalg.solve(mass, force)

fig, ax = plt.subplots()
x = np.linspace(0, 1, num = 1000)
sns.lineplot(y = psi, x = nodes, ax = ax)
sns.lineplot(y = x * (x**2 - 3*x + 3)/6, x = x, ax = ax)
plt.show()