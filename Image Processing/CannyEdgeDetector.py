import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def gaussFilter(img_in, ksize, sigma):
    """
    Apply Gaussian filter to the image
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # Generate 1D Gaussian kernel
    x = np.arange(ksize) - (ksize - 1) / 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # Create 2D kernel by outer product
    kernel = np.outer(kernel_1d, kernel_1d)
    
    # Apply convolution
    filtered = convolve(img_in, kernel)
    
    return kernel, filtered.astype(int)


def sobel(img_in):
    """
    Apply sobel filters to the input image
    Note: scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # Sobel kernels (scipy.ndimage.convolve flips them internally)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # Apply convolution
    gx = convolve(img_in, sobel_x).astype(int)
    gy = convolve(img_in, sobel_y).astype(int)
    
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    Calculate the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in y direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # Calculate gradient magnitude
    g = np.sqrt(gx**2 + gy**2).astype(int)
    
    # Calculate gradient direction in radians
    theta = np.arctan2(gy, gx)
    
    return g, theta


def convertAngle(angle):
    """
    Find nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # Convert to degrees
    angle_deg = np.degrees(angle) % 180  # Convert to [0, 180)
    
    # Find nearest matching angle
    if angle_deg < 22.5 or angle_deg >= 157.5:
        return 0
    elif angle_deg < 67.5:
        return 45
    elif angle_deg < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    Apply non-maximum suppression
    :param g: gradient magnitude (np.ndarray)
    :param theta: gradient direction (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    h, w = g.shape
    max_sup = np.zeros_like(g)
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Convert angle and find nearest direction
            angle_deg = convertAngle(theta[y, x])
            
            # Get neighboring pixels based on gradient direction
            if angle_deg == 0:  # horizontal
                neighbors = [g[y, x-1], g[y, x+1]]
            elif angle_deg == 45:  # diagonal
                neighbors = [g[y+1, x-1], g[y-1, x+1]]
            elif angle_deg == 90:  # vertical
                neighbors = [g[y-1, x], g[y+1, x]]
            else:  # angle_deg == 135
                neighbors = [g[y-1, x-1], g[y+1, x+1]]
            
            # Keep pixel if it's local maximum
            if g[y, x] >= max(neighbors):
                max_sup[y, x] = g[y, x]
    
    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    height, width = max_sup.shape
    
    # First pass: classify pixels
    thresholding = np.zeros_like(max_sup)
    for y in range(height):
        for x in range(width):
            if max_sup[y, x] <= t_low:
                thresholding[y, x] = 0
            elif max_sup[y, x] <= t_high:
                thresholding[y, x] = 1
            else:
                thresholding[y, x] = 2
    
    # Initialize result
    result = np.zeros_like(max_sup)
    
    # Second pass: set strong edges to 255
    for y in range(height):
        for x in range(width):
            if thresholding[y, x] == 2:  # Strong edge
                result[y, x] = 255
    
    # Third pass: connect weak edges to strong edges
    for y in range(height):
        for x in range(width):
            if thresholding[y, x] == 2:  # Strong edge
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            thresholding[ny, nx] == 1):  # Weak edge
                            result[ny, nx] = 255
    
    return result


def canny(img):
    # Apply gaussian filter
    kernel, gauss = gaussFilter(img, 5, 2)

    # Apply sobel filters
    gx, gy = sobel(gauss)

    # Show gradient images
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar(im1, shrink=0.8)
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar(im2, shrink=0.8)
    plt.tight_layout()
    plt.show()

    # Calculate gradient magnitude and direction
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar(im1, shrink=0.8)
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(theta)
    plt.title('theta')
    plt.colorbar(im2, shrink=0.8)
    plt.tight_layout()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.figure(figsize=(8, 6))
    im = plt.imshow(maxS_img, 'gray')
    plt.title('Maximum Suppression')
    plt.colorbar(im, shrink=0.8)
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result


if __name__ == "__main__":
    from PIL import Image
    
    # Load test image
    img = Image.open('data/contrast.jpg').convert('L')
    img = np.array(img)
    
    if img is not None:
        print("Image loaded successfully!")
        print(f"Image shape: {img.shape}")
        
        # Run Canny edge detection
        print("Running Canny edge detection...")
        result = canny(img)
        
        # Show final result
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result, cmap='gray')
        plt.title('Canny Edge Detection Result')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print("Canny edge detection completed!")
        print(f"Result shape: {result.shape}")
    else:
        print("Failed to load image!")
