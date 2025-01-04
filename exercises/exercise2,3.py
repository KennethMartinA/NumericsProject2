# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 00:23:32 2024

@author: kam224
"""
import numpy as np
import matplotlib.pyplot as plt
from finiteElement.Space import Space
from mpl_toolkits.mplot3d import Axes3D


numberElements = 30


nodes, IEN, ID, boundaries = Space.generate_2d_grid(numberElements)

space = Space(nodes, IEN, boundaries)

def sourceFunc(x):
    x1 = x[0]
    x2 = x[1]
    return 2*x1*(x1-2)*(3*x2**2-3*x2+0.5) + x2**2 * (x2-1)**2
def sourceFunc1(x):
    x1 = np.sin(x[0])
    x2 = np.sin(x[1])
    return x1*x2

psi = space.solve(sourceFunc1)




ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0, 1, numberElements + 1), np.linspace(0, 1, numberElements + 1))
ax.plot_surface(X, Y, psi.reshape(X.shape))
analytic = X * (1 - X/2) * Y ** 2 * (1 - Y)**2
ax.plot_surface(X, Y, analytic)  
plt.show()
