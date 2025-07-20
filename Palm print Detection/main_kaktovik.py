import numpy as np
import matplotlib.pyplot as plt
import cv2

from FourierTransform import calcuateFourierParameters
from DistanceMeasure import calculate_Theta_Distance, mseDistance
from kaktovikAlignmentSimple import simpleAlignment


# --- Parameters for frequency features ---
k = 6                # Number of angular sectors (for θ feature)
samplingSize = 128   # Image size after alignment

# --- Load grayscale images ---
img1 = cv2.imread('img/kaktovik-008_rot_0.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img/kaktovik-012_rot_0.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('img/kaktovik-012_rot_2.jpg', cv2.IMREAD_GRAYSCALE)



# --- Align and normalize images ---
img1_aligned = simpleAlignment(img1)
img2_aligned = simpleAlignment(img2)
img3_aligned = simpleAlignment(img3)

# --- Extract θ features (angular frequency distribution) ---
_, ThetaX = calcuateFourierParameters(img1_aligned, k, samplingSize)
_, ThetaY = calcuateFourierParameters(img2_aligned, k, samplingSize)
_, ThetaZ = calcuateFourierParameters(img3_aligned, k, samplingSize)



# --- Print comparison results ---
def printResult(name, DTheta, DTmpl):
    print(f"{name}")
    print(f"DTheta  (angular freq): {np.round(DTheta, 2)}")
    print(f"MSE distance: {np.round(DTmpl, 2)}")
    print()

print("Comparison results (lower = more similar):")
printResult("1 vs 1", calculate_Theta_Distance(ThetaX, ThetaX), mseDistance(img1_aligned, img1_aligned))
printResult("2 vs 1", calculate_Theta_Distance(ThetaY, ThetaX), mseDistance(img2_aligned, img1_aligned))
printResult("1 vs 2", calculate_Theta_Distance(ThetaX, ThetaY), mseDistance(img1_aligned, img2_aligned))
printResult("1 vs 3", calculate_Theta_Distance(ThetaX, ThetaZ), mseDistance(img1_aligned, img3_aligned))
printResult("2 vs 3", calculate_Theta_Distance(ThetaY, ThetaZ), mseDistance(img2_aligned, img3_aligned))

# --- Visualization ---
fig, axs = plt.subplots(2, 3)
axs[0,0].imshow(img1, cmap='gray'); axs[0,0].set_title('Input 1'); axs[0,0].axis('off')
axs[0,1].imshow(img2, cmap='gray'); axs[0,1].set_title('Input 2'); axs[0,1].axis('off')
axs[0,2].imshow(img3, cmap='gray'); axs[0,2].set_title('Input 3'); axs[0,2].axis('off')
axs[1,0].imshow(img1_aligned, cmap='gray'); axs[1,0].set_title('Aligned 1'); axs[1,0].axis('off')
axs[1,1].imshow(img2_aligned, cmap='gray'); axs[1,1].set_title('Aligned 2'); axs[1,1].axis('off')
axs[1,2].imshow(img3_aligned, cmap='gray'); axs[1,2].set_title('Aligned 3'); axs[1,2].axis('off')
plt.tight_layout()
plt.show()
