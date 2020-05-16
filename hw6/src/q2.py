# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from matplotlib import pyplot as plt
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    u, sv, vh = np.linalg.svd(I, full_matrices=False)
    sv = sv[:3]
    sigma = np.diag(sv)
    sigma = np.sqrt(sigma)

    B = np.dot(sigma, vh[:3, :])
    L = np.dot(u[:, :3], sigma).T

    # B = np.dot(sigma, vh[:3, :])
    # L = u[:, :3].T

    return B, L

def bas_relief(B, m=1, n=1, l=1):
    """
    Question 2 (e)

    Estimate new pseudonormals with bas-relief.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    m : float
        mu value
    n : float
        nu value
    l : float
        lambda value

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    g = np.eye(3)
    g[2, 0] = m
    g[2, 1] = n
    g[2, 2] = l
    B = np.dot(np.linalg.inv(g).T, B)
    return B

if __name__ == "__main__":

    I, L_0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    B = enforceIntegrability(B, s, 3)
    B = bas_relief(B, 0, 0, 2)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    print(L)
    print(L_0)
    surface = estimateShape(normals, s)
    plotSurface(surface)
