# Computer Vision & Machine Learning Project Collection

This repository contains various computer vision and machine learning projects organized into different folders. Each folder focuses on a specific area of image processing or computer vision.

## Project Structure

### Face Detection

A complete face recognition system using eigenfaces (Principal Component Analysis for face recognition).

**What it does:**

- Real-time face recognition using webcam to detect and identify faces
- Dataset testing mode for evaluating performance on face image collections
- Eigenfaces algorithm implementation with face database creation and SVD
- Feature extraction and classification to identify different people

**Training data includes multiple face images of different people**

### Image Processing

Various image processing algorithms implemented from scratch.

**What it does:**

- Canny edge detection with Gaussian filtering, Sobel gradients, and hysteresis thresholding
- Custom convolution operations using Gaussian kernels without built-in functions
- Morphological operations for binary images including erosion, dilation, opening and closing

**Sample images available for testing**

### Palm Print Detection

Palmprint recognition system using frequency domain analysis.

**What it does:**

- Automatic palmprint alignment from hand images using different methods
- Fourier transform analysis to extract frequency-based features from palmprints
- Distance measurement comparison between palmprints for identification
- Visualization tools for frequency domain results and alignment outcomes

**Includes sample hand images and research documentation**

### Preprocessing

Fundamental image preprocessing techniques for enhancing and preparing images.

**What it does:**

- Histogram equalization for contrast enhancement using custom computation and CDF calculation
- Otsu's automatic thresholding for binary image segmentation
- Noise generation and addition including Gaussian noise and salt & pepper noise
- Interactive Jupyter notebooks for hands-on exploration of preprocessing techniques

**Includes sample images for experiments**

### Running the Projects

1. Face Recognition: Real-time webcam mode or dataset testing mode
2. Image Processing: Independent algorithm testing and experimentation
3. Palmprint Recognition: Automatic palmprint analysis and comparison
4. Preprocessing: Interactive notebooks and direct algorithm application

Each implementation focuses on understanding algorithms from the ground up rather than using built-in functions.
