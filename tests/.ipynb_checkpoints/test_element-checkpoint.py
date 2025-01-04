import numpy as np
import pytest
from  finiteElement.Space import Element

STANDARD_NODES = np.array([[0,3,3],[0,0,3]])

def createStandardElement():
    return Element(STANDARD_NODES)

def testValidatePsi():
    element = createStandardElement()

    invalidPsis = [(-1,0), (0,-1), (2,0), (0,2), (0.75, 0.26)]

    for psi in invalidPsis:
        with pytest.raises(ValueError):
            element.globalCords(psi)

def testRefShapeValue():
    element = createStandardElement()
    psis = [0,0,0,0]
    psis[0] = (0,0)
    psis[1] = (1,0)
    psis[2] = (0,1)
    psis[3] = (0.5, 0.5)
    expectedVals = [[1,0,0], [0,1,0], [0,0,1], [0,0.5,0.5]]
    actualVals = [0,0,0,0]
    for i in range(4):
        psi = psis[i]
        actualVals[i] = element.refShapeValue(psi)
        assert np.isclose(np.array(actualVals[i]), expectedVals[i]).all()

def testRefShapeDiff():
    element = createStandardElement()
    psi = np.array([0,0])
    expected = np.array([[-1,1,0], [-1,0,1]]).T
    actual = element.refShapeDiff(psi)
    assert np.isclose(actual, expected).all()

def testGlobalCoords():
    element = createStandardElement()
    psis = [0,0,0,0]
    psis[0] = (0,0)
    psis[1] = (1,0)
    psis[2] = (0,1)
    psis[3] = (0.25, 0.25)
    expectedVals = [[0,0], [3,0], [3,3], [1.5, 0.75]]
    actualVals = [0,0,0,0]
    for i in range(4):
        psi = psis[i]
        actualVals[i] = element.globalCords(psi)
        assert np.isclose(np.array(actualVals[i]), expectedVals[i]).all()

def testLocalJacobian():
    element = createStandardElement()
    expectedMat = np.array([[3, 3], [0, 3]])
    actualMat = element.localJacobian((0,0))
    assert np.isclose(expectedMat, actualMat).all()

def testDetJacobian():
    element = createStandardElement()
    expectedVal = 9.
    actualVal = element.detJacobian((0,0))
    assert (np.isclose(actualVal, expectedVal)).all()

def testGlobalShapeDiff():
    element = createStandardElement()
    expectedVal = np.array([[-1./3, 0], [1./3, -1./3], [0, 1./3]])
    actualVal = element.globalShapeDiff((0,0))
    print(expectedVal)
    print(actualVal)
    assert (np.isclose(actualVal, expectedVal)).all()

def testLocalIntegral():
    element = createStandardElement()
    expectedVal = 1./24
    actualVal = element.localIntegral(lambda x : x[0] * x[1])
    assert np.isclose(actualVal, expectedVal)

def testGlobalIntegral():
    element = createStandardElement()
    expectedVal = 4.5
    actualVal = element.globalIntegral(lambda x : 1)
    assert np.isclose(actualVal, expectedVal)

def testStiffnessMatrix():
    element = createStandardElement()
    stiffnessMatrix = element.localStiffness()
    print(stiffnessMatrix)
    assert True

def testLocalForce():
    element = createStandardElement()
    forceVector = element.localForce(lambda x : 1)
    print(forceVector)
    assert True