import numpy as np
from constants import SOUTHAMPTON_LOC
import argparse

def gaussianSource(psi):
    x = psi[0]
    y = psi[1]    
    return np.exp(-0.5*((x-SOUTHAMPTON_LOC[0])**2 + (y-SOUTHAMPTON_LOC[1])**2) / 1000**2)/1000

def gaussianWeights(psi, loc, var = 10000):  
    return np.exp(-np.sum(np.power(psi - loc, 2), axis = 1)/ var**2)

def createMainArgParse():
    description = """
    This runs the main finite element model for advection diffusion on a provided triangular grid.
    """
    parser = argparse.ArgumentParser("finiteElementModel")
    parser.add_argument("--scale", help = "scale of triangular mesh",
                        choices = ["100k", "50k", "25k", "12_5k", "6_25k"], default = "6_25k")
    parser.add_argument("--plotNodalWeights", 
                       help = "if true, will create a plot of nodal weights and save", 
                       action=argparse.BooleanOptionalAction, default = True)
    parser.add_argument("--overrideFilepath", help = "if given, will override the default" + 
                        "filepath to mesh grids", default = "")
    parser.add_argument("--fileIdentifier", help = "if given, will append the identifier to the end"+
                        " of files created by the script", default = "")
    parser.add_argument("--totalHours", help = "if given, will override the default" + 
                        "total number of hours to run for",  default = 16, type = int)
    parser.add_argument("--dt", help = "if given, will override the default" + 
                        "timestep size",  default = 10, type = int)
    parser.add_argument("--diff", help = "if given, will override the default" + 
                    "diffusion coefficient", default = 10e-6, type = float)
    parser.add_argument("--plotHourlyPsi", 
                       help = "if true, will create a plot and save for each resultant psi at each" + 
                       "hour given", 
                       action=argparse.BooleanOptionalAction, default = False)
    
    parser.add_argument("--useFilter", 
                    help = "if true, will also use a temporal filter by averaging values of psi over 20" +
                    "locations in time. Recommended.",
                    action=argparse.BooleanOptionalAction, default = False)

    return parser



class psiFilter:
    """Temporal fitler for Psi
    """

    def __init__(self, psi_shape,width = 20):
        self.width = width
        self.psis = np.zeros((width, psi_shape[0]))
        self.counter = 0

    def append(self, psi):
        self.psis[self.counter, :] = psi
        self.counter += 1
        self.counter = self.counter % self.width

    def getPsis(self):
        return self.psis.mean(axis = 0)

