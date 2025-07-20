'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # Calculate mean absolute difference
    distance = np.mean(np.abs(Rx - Ry))
    return distance


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # Calculate normalized distance based on vector norms
    lxx = np.sum(Thetax ** 2)
    lyy = np.sum(Thetay ** 2)
    lxy = np.sum(Thetax * Thetay)
    
    if lxx + lyy == 0:
        return 0.0
    
    # Calculate normalized distance
    distance = np.sqrt((lxx + lyy - 2 * lxy) / (lxx + lyy)) * 84.5
    return distance


# Mean Squared Error (MSE) as additional comparison
def mseDistance(imgA, imgB):
    """
    Computes the mean squared difference between two equally sized images.
    param imgA: First image.
    param imgB: Second image.
    return: Mean squared error between the two images.
    Hint: 0 means identical, higher values indicate more differences.
    """
    # Ensure images have the same shape
    if imgA.shape != imgB.shape:
        # Resize images to same size if different
        min_height = min(imgA.shape[0], imgB.shape[0])
        min_width = min(imgA.shape[1], imgB.shape[1])
        imgA = imgA[:min_height, :min_width]
        imgB = imgB[:min_height, :min_width]
    
    # Calculate mean squared error
    mse = np.mean((imgA.astype(float) - imgB.astype(float)) ** 2)
    return mse