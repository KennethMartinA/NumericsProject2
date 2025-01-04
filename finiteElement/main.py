# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 00:23:32 2024

@author: kam224
"""
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from utils import *
from finiteElement.SpatialInterpolator import SpatialInterpolator
from finiteElement.Space import sparseAdvectionDiffSpace
import pickle
import argparse

parser = createMainArgParse()

plot = False
filepath = DEFAULT_FILEPATH

args = parser.parse_args()
scale = args.scale + ".txt"

if args.overrideFilepath != "":
    DEFAULT_FILEPATH = args.overrideFilepath

nodes = np.loadtxt(filepath + 'nodes_' + scale)
IEN = np.loadtxt(filepath + 'IEN_' + scale, dtype=np.int64)
#correction for IEN ordering
IEN = IEN[:,-1::-1]
boundaries = np.loadtxt(filepath + 'bdry_' + scale, 
                            dtype=np.int64)

with open("finiteElement/model.pkl", "rb") as f:
    interp = pickle.load(f)
print("Loaded Successfully")
df = interp.interpolate(nodes.T,[0])[["horizontal_wind_speed", "vertical_wind_speed"]]
initialVelocityMatrix = df[["horizontal_wind_speed", "vertical_wind_speed"]].to_numpy()

#initialize the weights for the weights mask with which Psi at reading is estimated with
weights = gaussianWeights(nodes, READING_LOC)
weights = weights/np.sum(weights)
#weights[weights < 10e-8] = 0

if args.plotNodalWeights == True:
    plt.scatter(nodes.T[0], nodes.T[1], c = weights)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.axis(PLOTTING_WINDOW)
    plt.savefig("Images/nodalWeights.png")
    plt.clf()

#initialize the space
space = sparseAdvectionDiffSpace(nodes, IEN, boundaries, initialVelocityMatrix, args.diff)
space.assemble(gaussianSource)

with open("results/results" + args.fileIdentifier + ".csv", "w") as f:
    f.write("time, concentration\n")

timestepsPerHour = int(3600/args.dt)
 
space.timestep(0.1)
psi = space.getPsi()
if args.useFilter:
    psiFilter = psiFilter(psi.shape)
    psiFilter.append(psi)

for hour in range(16):
    for i in range(timestepsPerHour - 1):
        if args.useFilter:
            psiFilter.append(psi)
        space.timestep(args.dt)
    df = interp.interpolate(nodes.T,[space.cur_t/3600])[["horizontal_wind_speed", "vertical_wind_speed"]]
    updatedVelocityMatrix = df.to_numpy()
    space.timestep(args.dt, updatedVelocityMatrix)
    print(space.cur_t)
    psi = space.getPsi()
    if args.useFilter:
        psiFilter.append(psi)
    with open("results/results" + args.fileIdentifier + ".csv", "a") as f:
        if args.useFilter:
            psiFilter.append(psi)
            psi = psiFilter.getPsis()
        normalized_psi = (psi)/np.sum(psi)
        normalized_psi[normalized_psi < 10e-8] = 0
        f.write(str(space.cur_t) + ", " + str(np.sum(weights  * normalized_psi)) + "\n")
    if args.plotHourlyPsi:
        plt.cla()
        plt.tripcolor(nodes[:,0], nodes[:,1], psi, triangles = IEN,shading = "flat")
        plt.axis('equal')
        plt.axis([200000,600000, 0, 400000])
        plt.savefig("Images/results " + str(int(space.cur_t/3600)) + args.fileIdentifier + ".png")


