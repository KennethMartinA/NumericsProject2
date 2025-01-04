import numpy as np
class Element:
    """Basic element of the finite element solver. Represents one triangle in the mesh.

    Raises:
        ValueError: If given invalid 2-vector on its internal reference triangle.
    """
    nodes = np.zeros((2,3))
    
    def validatePsi(func):
        def wrapper(self, psi):
            if psi[1] < 0 or psi[0] < 0 or psi[1] > 1 - psi[0] or psi[0] > 1:
                raise ValueError("Invalid coordinates for psi")
            else:
                return func(self, psi)
        return wrapper

    def __init__(self, nodes):
        """
        Args:
            nodes : a 2-vector, first row is x values, 2nd row is y values
        """
        self.nodes = nodes
        assert nodes.shape == (2,3)
    
    @validatePsi
    def refShapeValue(self, psi):
        # 3-vector, each entry is the shape function's value at psi
        arr = np.zeros(3)
        arr[0] = 1 - np.sum(psi)
        arr[1:] = psi
        return arr
    
    @validatePsi
    def refShapeDiff(self, psi):
        #derivative of shape functions for each shape function and for x or y
        arr = np.array([[-1, 1, 0],[-1, 0, 1]])
        return arr.T
    
    @validatePsi
    def globalCords(self, psi):
        #convert from reference to global coordinates
        refCoords = self.refShapeValue(psi)
        return np.matmul(refCoords, self.nodes.T)
    
    @validatePsi
    def localJacobian(self, psi):
        #jacobian matrix
        shapeDeriv = self.refShapeDiff(psi)
        jacobian = np.matmul(self.nodes, shapeDeriv)
        return jacobian
    
    @validatePsi
    def detJacobian(self, psi):
        #determinant of the jacobian
        jacobian = self.localJacobian(psi)
        return np.linalg.det(jacobian)
    
    @validatePsi
    def globalShapeDiff(self, psi):
        #Shape derivative in terms of global coordinates
        jacobian = self.localJacobian(psi)
        shapeDiffs = self.refShapeDiff(psi)
        inv_jacobian = np.linalg.inv(jacobian)
        return np.matmul(shapeDiffs, inv_jacobian)

    def localIntegral(self, integrandFunc):
        #gaussian quadrature, approximates an integral of a function over a triangle
        locs = 1/6 * np.array([[1,1],[1,4],[4,1]])
        result = np.zeros(3)
        for i in range(len(result)):
            result[i] = integrandFunc(locs[i])
        return np.sum(result)/6.

    def globalIntegral(self, integrandFunc):
        #calls localIntegral, but using a jacobian and converting local coordinates to global
        jacobian = self.detJacobian((0,0))
        def globalCordWrapper(function):
            def inner(psi):
                psi_new = self.globalCords(psi)
                result = function(psi_new)
                return result
            return inner
        
        transformFunc = globalCordWrapper(integrandFunc)
        integrand = self.localIntegral(transformFunc)
        return jacobian * integrand

    
    def localStiffness(self):
        #Calculates the steady-state diffusion stiffness matrix
        globalDiff = self.globalShapeDiff((0,0))
        globalDiff_matrix = np.matmul(globalDiff, globalDiff.T)
        stiffnessMatrix = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                stiffnessMatrix[i][j] = self.globalIntegral(lambda x : globalDiff_matrix[i][j])
        return stiffnessMatrix
    
    def localForce(self, sourceFunc):
        #local force vector
        forceVector = np.zeros(3)
        shapeFunction = self.refShapeValue
        jacobian = self.detJacobian((0,0))
        for i in range(3):
            def transFunc(psi):
                return shapeFunction(psi)[i] * jacobian * sourceFunc(self.globalCords(psi))
            forceVector[i] = self.localIntegral(transFunc)

        return forceVector
    
        
    
class AdvecDiffElement(Element):
    def __init__(self, nodes, diff, velocityMatrix):
        """Initialize the element.

        Args:
            nodes : array of the element's global node locations
            diff (float): diffusion coefficient
            velocityMatrix (np.array((3,2))): nodal velocity matrix, representing x,y velocity
            at each nodal point
        """
        self.diff = diff
        self.velocity_x, self.velocity_y = velocityMatrix.mean(axis = 0)
        self.nodes = nodes
        assert nodes.shape == (2,3)

    def setVelocity(self, velocityMatrix):
        self.velocity_x, self.velocity_y = velocityMatrix.mean(axis = 0)

    def localStiffness(self):
        """Calculates and stores the x-component of the local advection stiffnness,
        y component, and the diffusion. Then it adds them and reutrn them.

        """
        globalDiff = self.globalShapeDiff((0,0))
        globalDiff_matrix = np.matmul(globalDiff, globalDiff.T)
        jacobian = self.detJacobian((0,0))
        stiffnessMatrix = np.zeros((3,3))

        self.advecMatrix_x = np.zeros((3,3))
        self.advecMatrix_y = np.zeros((3,3))
        self.diffMatrix = np.zeros((3,3))

        for i in range(3):
            for j in range(3):

                self.diffMatrix[i][j] = (self.globalIntegral(lambda x : self.diff * globalDiff_matrix[i][j]))
                stiffnessMatrix[i][j] = self.diffMatrix[i][j]

                def transFunc(psi):
                    result =  ((self.refShapeValue(psi)[i]) * jacobian * 
                        ( globalDiff[j])[0])
                    return result
                    
                advecIntegralx = self.velocity_x * self.localIntegral(transFunc)
                self.advecMatrix_x[i][j] =   advecIntegralx

                def transFunc(psi):
                    result =  ((self.refShapeValue(psi)[i]) * jacobian * 
                        ( globalDiff[j])[1])
                    return result
                advecIntegraly = self.velocity_y * self.localIntegral(transFunc)
                self.advecMatrix_y[i][j] = advecIntegraly
                
                stiffnessMatrix[i][j] -= (advecIntegralx + advecIntegraly)

        return stiffnessMatrix
    
    def localForce(self, sourceFunc):
        """Calculates local force of the element

        Args:
            sourceFunc : A python function that should take a 2 vector as argument,
            and return a float

        Returns:
            Force vector
        """
        globalDiff = self.globalShapeDiff((0,0))
        forceVector = np.zeros(3)
        shapeFunction = self.refShapeValue
        jacobian = self.detJacobian((0,0))
        for i in range(3):
            def transFunc(psi):
                return (shapeFunction(psi)[i]) * jacobian * sourceFunc(self.globalCords(psi))
            forceVector[i] = self.localIntegral(transFunc)

        return forceVector
    
    def updateStiffness(self, newVelocityMatrix):
        """This function updates the stiffness based on cached values.

        Specifically, localStiffness needs to be called the first time, calculcated as
        Stiff = Advec_x + Advec_y + Diff, and storing the latter 3 matrices.
        This function uses an updated velocity matrix to recalculcate Stiff
        as v_x_new / v_x * Advec_x + v_y_new / v_y * Advec_y + Diff 


        Returns
            Local stiffness
        """
        new_vx, new_vy = newVelocityMatrix.mean(axis = 0)
        stiffnessMatrix = np.zeros((3,3))
        stiffnessMatrix += self.diffMatrix
        stiffnessMatrix -= (new_vx/self.velocity_x * self.advecMatrix_x + 
            new_vy/self.velocity_y * self.advecMatrix_y) 
        return stiffnessMatrix    
        
    def localMass(self):
        #Calculate the local mass matrix
        massMatrix = np.zeros((3,3))
        jacobian = self.detJacobian((0,0))
        
        for i in range(3):
            for j in range(3):
                massMatrix[i][j] = self.localIntegral(lambda x : self.refShapeValue(x)[i] * jacobian
                                                       * self.refShapeValue(x)[j])
        return massMatrix
    
    def triangleArea(self):
        return self.globalIntegral(lambda x : 1)
    

