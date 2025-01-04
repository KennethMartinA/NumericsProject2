# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:55:37 2024

Based on code written by Professor Ian Hawke for the MFC Numerics Course Part 2
"""
import numpy as np 
from scipy import sparse
from finiteElement.Element import *
import time

class Space:
    """This class manages the spatial mesh and its elements, along with the solver.

    This class specifically solves the steady state diffusion PDE on a 2D triangular domain.
    """

    def __init__(self, nodes, IEN, boundaries, ID = 0):
        """initialize space with correct nodes, IEN, boundaries and ID

        Args:
            nodes : array of nodal locations
            IEN : array of arrays, where each subarray is a triangular element's nodes 
            boundaries: boundary nodes
            ID: array containing the interior nodes
        """
        if type(ID) is int:
            self.ID = np.zeros(len(nodes), dtype=np.int64)
            eq_count = 0
            for i in range(len(nodes[:, 1])):
                if i in boundaries:
                    self.ID[i] = -1
                else:
                    self.ID[i] = eq_count
                    eq_count += 1
        else:
            self.ID = ID
        self.nodes = nodes
        self.IEN = IEN
        self.boundaries = boundaries
        self.Elements = []

        #array of Element objects
        for i in range(IEN.shape[0]):
            self.Elements.append(Element(np.array(nodes[IEN[i]]).T))

        numEquations = np.max(self.ID) + 1
        self.stiffnessMatrix = np.zeros((numEquations, numEquations))
        self.forceVector = np.zeros(numEquations)

        #here we construct the location matrix, linking local elemental nodes and global nodal locations
        self.LM = np.zeros_like(IEN.T)
        for e in range(len(self.Elements)):
            for a in range(3):
                self.LM[a,e] = self.ID[IEN[e,a]]

    def locationMatrix(self, elementIndex, nodeIndex):
        "a function for LM, to be more explicit"
        return self.LM[nodeIndex, elementIndex]
    
    def solve(self, sourceFunc):
        """Solves the steady state diffusion problem

        Args:
            sourceFunc : A python function that takes a 2-vector as argument 
            and outputs a single number

        Returns:
            a np column vector of floats matching nodal column shape, representing the
            solved Psi values at nodes
        """
        for i in range(len(self.Elements)):
            element = self.Elements[i]
            localStiffness = element.localStiffness()
            localForce = element.localForce(sourceFunc)
            for a in range(3):
                A = self.locationMatrix(i, a)
                for b in range(3):
                    B = self.locationMatrix(i, b)
                    if (A >= 0) and (B >= 0):
                        self.stiffnessMatrix[A, B] += localStiffness[a, b]
                if (A >= 0):
                    self.forceVector[A] += localForce[a]
        # Solve the equation
        Psi_interior = np.linalg.solve(self.stiffnessMatrix, self.forceVector)
        Psi_A = np.zeros(self.nodes.shape[0])
        for n in range(self.nodes.shape[0]):
            if self.ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
                Psi_A[n] = Psi_interior[self.ID[n]]
        self.Psi = Psi_A
        return Psi_A
    def getPsi(self):
        return self.Psi
    @staticmethod
    #method by Professor Ian Hawke to create a 2D domain for testing
    def generate_2d_grid(Nx):
        Nnodes = Nx+1
        x = np.linspace(0, 1, Nnodes)
        y = np.linspace(0, 1, Nnodes)
        X, Y = np.meshgrid(x,y)
        nodes = np.zeros((Nnodes**2,2))
        nodes[:,0] = X.ravel()
        nodes[:,1] = Y.ravel()
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # Will hold the boundary values
        n_eq = 0
        for nID in range(len(nodes)):
            if np.allclose(nodes[nID, 0], 0):
                ID[nID] = -1
                boundaries[nID] = 0  # Dirichlet BC
            else:
                ID[nID] = n_eq
                n_eq += 1
                if ((np.allclose(nodes[nID, 1], 0)) or 
                    (np.allclose(nodes[nID, 0], 1)) or 
                    (np.allclose(nodes[nID, 1], 1)) ):
                    boundaries[nID] = 0 # Neumann BC
        IEN = np.zeros((2*Nx**2, 3), dtype=np.int64)
        for i in range(Nx):
            for j in range(Nx):
                IEN[2*i+2*j*Nx  , :] = (i+j*Nnodes, 
                                        i+1+j*Nnodes, 
                                        i+(j+1)*Nnodes)
                IEN[2*i+1+2*j*Nx, :] = (i+1+j*Nnodes, 
                                        i+1+(j+1)*Nnodes, 
                                        i+(j+1)*Nnodes)
        return nodes, IEN, ID, boundaries

class sparseAdvectionDiffSpace(Space):
    
    def __init__(self, nodes, IEN, boundaries, nodalVelocities, 
                 diff_coeff : np.float64, ID = 0):
        """Similar to parent, except it is meant for the time-dependent
        advection diffusion equation. Manages the grid, the timesteps, and updates.
        Implemented using scipy sparse matrices for more efficiency

        Args:
            nodalVelocities : array matching shapes of nodes, containing x and y component of velocity
            at each point
            diff_coeff : corresponds to the diffusion coefficient
        """
        #check for ID being given already
        if type(ID) is int:
            self.ID = np.zeros(len(nodes), dtype=np.int64)
            eq_count = 0
            for i in range(len(nodes[:, 1])):
                if i in boundaries:
                    self.ID[i] = -1
                else:
                    self.ID[i] = eq_count
                    eq_count += 1
        else:
            self.ID = ID
        self.nodes = nodes
        self.IEN = IEN
        self.boundaries = boundaries
        self.Elements = []
        for i in range(IEN.shape[0]):
            self.Elements.append(AdvecDiffElement(np.array(nodes[IEN[i]]).T, 
                                                  diff_coeff, 
                                                  nodalVelocities[IEN[i]])
                                                  )
        self.numEquations = np.max(self.ID) + 1
        #set following sparse only after assembly
        self.stiffnessMatrix = np.zeros((self.numEquations, self.numEquations))
        self.massMatrix = np.zeros((self.numEquations, self.numEquations))
        self.forceVector = np.zeros(self.numEquations)

        #not sparse yet
        self.Psi = sparse.csc_array(np.zeros((self.numEquations, 1)))
        self.full_Psi = sparse.csc_array(np.zeros((self.nodes.shape[0], 1)))

        self.LM = np.zeros_like(IEN.T)
        for e in range(len(self.Elements)):
            for a in range(3):
                self.LM[a,e] = self.ID[IEN[e,a]]

    def assemble(self, sourceFunc):
        """assembles mass, stiffnes and force vectors

        Based on code by Professor Ian Hawke

        Args:
            sourceFunc : a python function that takes a 2 vector as argument
            and outputs a single number
        """
        self.cur_t = 0
        for i in range(len(self.Elements)):
            element = self.Elements[i]
            localStiffness = element.localStiffness()
            localMass = element.localMass()
            localForce = element.localForce(sourceFunc)
            for a in range(3):
                A = self.locationMatrix(i, a)
                for b in range(3):
                    B = self.locationMatrix(i, b)
                    if (A >= 0) and (B >= 0):
                        self.stiffnessMatrix[A, B] += localStiffness[a, b]
                        self.massMatrix[A, B] += localMass[a, b]
                if (A >= 0):
                    self.forceVector[A] += localForce[a]
        self.stiffnessMatrix = sparse.csr_matrix(self.stiffnessMatrix)
        self.massMatrix = sparse.csr_matrix(self.massMatrix)
        self.forceVector = sparse.csc_array(self.forceVector.reshape(
            self.forceVector.shape[0],1
        ))

    def timestep(self, dt, nodalVelocities = -1, print_time = False):
        """RK2 timesteps for the constructed matrices. dt = 100 works relatively well
        if nodalVelocities are not given, it simply uses the nodalVelocities given
        at initialization. Use case is for time-dependent velocity fields.
        Note that reassembly is expensive.

        RK2 step taken from Professor Ian Hawke's notes.

        Args:
            dt (int): temporal resolution
            nodalVelocities (optional): Expects a column of 2D vectors matching nodes. Defaults to -1.
            print_time (bool, optional): if true, will print time for profiling purposes. Defaults to False.
        """
        if print_time:
            init_time = time.time()
        if not(type(nodalVelocities) is int):
            self.reassembleStiffness(nodalVelocities)
            if print_time:
                print ("reassembly time: ", (time.time() - init_time))
        self.cur_t += dt
        # RK2 step 1
        dpsidt = sparse.linalg.spsolve(self.massMatrix, 
                                 self.modifyForce() * self.forceVector - self.stiffnessMatrix @ self.Psi)
        dpsidt = dpsidt.reshape((dpsidt.shape[0],1))
        Psi1 = self.Psi.copy()
        Psi1 += dt * dpsidt
        # RK2 step 2
        dpsidt = sparse.linalg.spsolve(self.massMatrix, 
                                  self.modifyForce() * self.forceVector - self.stiffnessMatrix @ self.Psi)
        dpsidt = dpsidt.reshape((dpsidt.shape[0],1))
        Psi1 = self.Psi.copy()
        self.Psi = (self.Psi + Psi1 + dt * dpsidt) / 2
        self.full_Psi[self.ID >= 0] = self.Psi
        if print_time:
            print ("total time: ", (time.time() - init_time))

    def modifyForce(self):
        """modifies force based on time elapsed
        """
        diagonalForcing = sparse.lil_matrix((self.numEquations, self.numEquations))
        diagonalForcing.setdiag(np.ones(self.numEquations) * np.exp(-(self.cur_t/(3600 * 8) - 0.5)))
        return diagonalForcing
    def reassembleStiffness(self, nodalVelocities):
        """Used by timestep to update the stiffness matrix using provided nodalVelocities.

        Args:
            nodalVelocities: a column vectors of velocity 2 vectors
        """
        self.stiffnessMatrix = np.zeros((self.numEquations, self.numEquations))
        for i in range(len(self.Elements)):
            element = self.Elements[i]
            localStiffness = element.updateStiffness(nodalVelocities[self.IEN[i]])
            for a in range(3):
                A = self.locationMatrix(i, a)
                for b in range(3):
                    B = self.locationMatrix(i, b)
                    if (A >= 0) and (B >= 0):
                        self.stiffnessMatrix[A, B] += localStiffness[a, b]            
        self.stiffnessMatrix = sparse.csr_matrix(self.stiffnessMatrix)

    def getPsi(self):
        #every space object defines its own getPsi method, to sanitize any output
        return self.full_Psi.toarray()[:,0]
