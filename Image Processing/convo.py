from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    """Create a Gaussian kernel for blurring"""
    # Calculate center position
    center = ksize // 2
    x, y = np.meshgrid(np.arange(ksize) - center, np.arange(ksize) - center)
    
    # Full Gaussian formula: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
    # But we'll normalize at the end anyway, so we can skip the 1/2πσ² factor
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize kernel so sum = 1 (this replaces the 1/2πσ² factor)
    kernel = kernel / np.sum(kernel)
    
    return kernel


def slow_convolve(arr, k):
    """Manual convolution implementation with zero padding"""
    if len(arr.shape) == 3:
        # Color image - apply to each channel
        result = np.zeros_like(arr)
        for ch in range(arr.shape[2]):
            result[:, :, ch] = slow_convolve(arr[:, :, ch], k)
        return result
    
    # Grayscale image
    img_h, img_w = arr.shape
    k_h, k_w = k.shape
    
    # Need to flip kernel for convolution (not correlation)
    flipped_k = np.flip(np.flip(k, 0), 1)
    
    # Padding size
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Zero padding
    padded_img = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Output array
    output = np.zeros((img_h, img_w))
    
    # Convolution loop
    for i in range(img_h):
        for j in range(img_w):
            # Get neighborhood
            neighborhood = padded_img[i:i+k_h, j:j+k_w]
            # Compute weighted sum
            output[i, j] = np.sum(neighborhood * flipped_k)
    
    return output


if __name__ == '__main__':
    # Kernel parameters
    kernel_size = 5
    sigma = 2.0
    
    # Create Gaussian blur kernel
    gaussian_kernel = make_kernel(kernel_size, sigma)
    
    # Load input image
    input_image = np.array(Image.open('data/input1.jpg'))
    
    print(f"Processing image: {input_image.shape}")
    print(f"Using {kernel_size}x{kernel_size} Gaussian kernel (σ={sigma})")
    print(f"Kernel sum: {np.sum(gaussian_kernel):.6f}")
    
    # Apply unsharp masking: result = input + (input - convolve(gaussian, input))
    blurred = slow_convolve(input_image.astype(np.float64), gaussian_kernel)
    unsharp_mask = input_image.astype(np.float64) - blurred
    enhanced = input_image.astype(np.float64) + unsharp_mask
    final_img = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Save the result
    Image.fromarray(final_img).save('data/res.jpg')
    print("Enhanced image saved as 'data/res.jpg'")
