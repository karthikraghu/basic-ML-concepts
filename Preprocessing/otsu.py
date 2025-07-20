import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_histogram(image: np.ndarray) -> np.ndarray:
    # ToDo: Compute a grayscale histogram with 256 bins.
    histogram = np.zeros(0)
    return histogram


def p_helper(prob: np.ndarray, theta: int) -> tuple[float, float]:
    # ToDo: Compute the class probabilities p0 and p1 for the current threshold theta.
    p0 = 0.0
    p1 = 0.0
    return p0, p1


def mu_helper(prob: np.ndarray, theta: int, p0: float, p1: float) -> tuple[float, float]:
    # ToDo: Compute the class means mu0 and mu1 for the current threshold theta.
    mu0 = 0.0
    mu1 = 0.0
    return mu0, mu1


def otsu_threshold(histogram: np.ndarray) -> int:
    # ToDo: Compute Otsu's threshold from a histogram using p_helper and mu_helper.
    # ToDo: Normalize the histogram to its probabilities (PDF).
    prob = histogram.astype(np.float64)

    # ToDo: Iterate over all possible thresholds, select the best one.
    # ToDo: Hint: Skip invalid splits (p0 == 0 or p1 == 0).
    max_variance = 0.0
    best_threshold = 0

    return best_threshold


def otsu_binarize(image: np.ndarray) -> tuple[np.ndarray, int]:
    # ToDo: Binarize the threshold image.
    # ToDo: Simply combine the existing functions.
    theta = 0
    new_image = np.zeros(0)
    return new_image, theta


def custom_binarization(image: np.ndarray, theta: int) -> tuple[np.ndarray, int]:
    # ToDo: Binarize the image with a custom value.
    new_image = np.where(image > theta, 255, 0).astype(np.uint8)
    return new_image, theta


if __name__ == '__main__':
    # Load grayscale image.
    loaded_image = cv2.imread('data/runes.png', cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError("Cannot load the image.")

    # Compute Otsu's binarization or perform a custom binarization. Comment out one of the options.
    # binarized_image, threshold = otsu_binarize(loaded_image)
    binarized_image, threshold = custom_binarization(loaded_image, 180)

    # Display the original and the binarized image next to each other.
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(loaded_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binarized_image, cmap='gray')
    plt.title(f"Otsu Binarization (t={threshold})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
