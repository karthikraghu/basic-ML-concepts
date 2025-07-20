'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    center_y, center_x = shape[0] // 2, shape[1] // 2
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    return int(y), int(x)


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    :param img:
    :return: Magnitude in Decibel
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_db = 20 * np.log10(np.abs(fshift) + 1e-10) # Add epsilon to avoid log(0)
    return magnitude_db


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features by summing pixel values in concentric bands.
    :param magnitude_spectrum: The input spectrum or image.
    :param k: The number of rings to extract, which corresponds to the number of features.
    :param sampling_steps: The number of angular samples to take within each band.
    :return: A feature vector of length k containing the sum of pixel values for each ring band.
    '''
    features = np.zeros(k, dtype=float)
    h, w = magnitude_spectrum.shape
    max_radius = min(h, w) // 2
    
    # This represents the width of each concentric ring band.
    band_width = max_radius / k

    # Iterate through each of the k ring bands.
    for i in range(k):
        ring_sum = 0.0
        
        # Define the start and end radius for the current band.
        r_start = i * band_width
        r_end = (i + 1) * band_width
        
        # Loop through all integer radii within the current band ("thick ring").
        for r in range(int(r_start), int(r_end) + 1):
            # Use np.linspace to sample the semicircle (0 to pi), inclusive of both endpoints.
            for theta in np.linspace(0, np.pi, sampling_steps, endpoint=True):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)

                # Check if the calculated coordinate is within the image bounds.
                if 0 <= y < h and 0 <= x < w:
                    ring_sum += magnitude_spectrum[y, x]
        
        # The feature is the sum of the pixel values.
        features[i] = ring_sum
            
    return features


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features by averaging pixel values in k sectors.
    :param magnitude_spectrum: The input spectrum or image.
    :param k: The number of fan-like sectors to extract.
    :param sampling_steps: The number of angular samples to take within each sector.
    :return: A feature vector of length k, containing the average pixel value per sector.
    """
    features = np.zeros(k, dtype=float)
    h, w = magnitude_spectrum.shape
    max_radius = min(h, w) // 2

    # The angular width of each sector. Divide the semicircle (pi) into k sectors.
    sector_angle_width = np.pi / k

    for i in range(k):
        fan_sum = 0.0
        count = 0
        
        angle_start = i * sector_angle_width

        # Take 'sampling_steps' discrete angular samples within this sector.
        for j in range(sampling_steps):
            theta = angle_start + (j / sampling_steps) * sector_angle_width

            # Sum the pixel magnitudes along a ray at this angle.
            for r in range(max_radius):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                
                if 0 <= y < h and 0 <= x < w:
                    fan_sum += magnitude_spectrum[y, x]
                    count += 1
        
        # Calculate the average value for the feature.
        if count > 0:
            features[i] = fan_sum / count
        else:
            features[i] = 0.0
            
    return features


def calcuateFourierParameters(img, k, sampling_steps):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R_features = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T_features = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    
    return R_features, T_features