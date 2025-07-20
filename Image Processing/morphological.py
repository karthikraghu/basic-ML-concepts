import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    """Extract the area around center pixel from padded image"""
    half = window_size // 2
    r1 = center_row - half
    r2 = center_row + half + 1
    c1 = center_col - half  
    c2 = center_col + half + 1
    
    return padded_image[r1:r2, c1:c2]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    """Add zero padding around the image"""
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    pad_size = se_size // 2
    padded_img = pad_image(image, pad_size)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_i = i + pad_size
            padded_j = j + pad_size
            window = extract_region(padded_img, padded_i, padded_j, se_size)
            
            # erosion: all SE positions must match
            if np.all((structuring_element == 0) | (window == structuring_element)):
                result[i, j] = 1
    
    return result


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    pad_size = se_size // 2
    padded_img = pad_image(image, pad_size)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_i = i + pad_size
            padded_j = j + pad_size
            window = extract_region(padded_img, padded_i, padded_j, se_size)
            
            # dilation: check if any foreground pixel in window overlaps with SE
            # If there's any white pixel where SE is 1, make output pixel white
            if np.any((structuring_element == 1) & (window == 1)):
                result[i, j] = 1
    
    return result


def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Opening: erosion followed by dilation"""
    result = input_image.copy()
    
    # First erode, then dilate
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)

    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Closing: dilation followed by erosion"""
    result = input_image.copy()
    
    # First dilate, then erode  
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)

    return result


def load_binary(filepath: str) -> np.ndarray:
    """Load image and convert to binary"""
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)
    # threshold at 128
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    """Save binary image to file"""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    """Display image with matplotlib"""
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # File paths
    erosion_input_path = 'data/erosion_image_raw.png'
    dilation_input_path = 'data/dilation_image_raw.png'
    erosion_output_path = 'data/erosion_output.png'
    dilation_output_path = 'data/dilation_output.png'

    # Load the binary images
    erosion_img = load_binary(erosion_input_path)
    dilation_img = load_binary(dilation_input_path)

    # 5x5 structuring element
    se = np.ones((5, 5), dtype=np.uint8)
    
    # Erosion - apply multiple times until circles separate
    eroded_img = erosion_img.copy()
    erosion_iterations = 0
    for i in range(10):  # max iterations
        eroded_img = erode_binary(eroded_img, se)
        erosion_iterations += 1
        print(f"Erosion iteration {erosion_iterations}")
        # stop after enough iterations 
        if erosion_iterations >= 4:  
            break
    
    save_binary(eroded_img, erosion_output_path)
    show_image(eroded_img, f"Erosion Result ({erosion_iterations} iterations)")

    # Dilation - apply multiple times until hole closes
    dilated_img = dilation_img.copy()
    dilation_iterations = 0
    for i in range(20):  # max iterations increased
        dilated_img = dilate_binary(dilated_img, se)
        dilation_iterations += 1
        print(f"Dilation iteration {dilation_iterations}")
        # stop after enough iterations
        if dilation_iterations >= 8:  # increased iterations
            break
            
    save_binary(dilated_img, dilation_output_path)
    show_image(dilated_img, f"Dilation Result ({dilation_iterations} iterations)")
