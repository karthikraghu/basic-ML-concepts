'''
Frequency Domain Visualization for Palmprint Analysis
Created on 16.07.2025

This script shows the Fourier transform results to understand
the frequency domain feature extraction process.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from FourierTransform import calculateMagnitudeSpectrum, extractRingFeatures, extractFanFeatures, polarToKart
from PalmprintAlignmentAutomatic import palmPrintAlignment

def visualize_frequency_domain():
    """
    Shows the frequency domain analysis of palmprint images
    """
    # Load and process images
    print("Loading palmprint images...")
    
    img1 = cv2.imread('img/Hand1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('img/Hand2.jpg', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('img/Hand3.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Align the images
    img1_aligned = palmPrintAlignment(img1)
    img2_aligned = palmPrintAlignment(img2)
    img3_aligned = palmPrintAlignment(img3)
    
    # Calculate magnitude spectrums
    spectrum1 = calculateMagnitudeSpectrum(img1_aligned)
    spectrum2 = calculateMagnitudeSpectrum(img2_aligned)
    spectrum3 = calculateMagnitudeSpectrum(img3_aligned)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Palmprint Frequency Domain Analysis', fontsize=16, fontweight='bold')
    
    # Row 1: Original aligned images
    axes[0, 0].imshow(img1_aligned, cmap='gray')
    axes[0, 0].set_title('Hand 1 - Aligned')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_aligned, cmap='gray')
    axes[0, 1].set_title('Hand 2 - Aligned')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img3_aligned, cmap='gray')
    axes[0, 2].set_title('Hand 3 - Aligned')
    axes[0, 2].axis('off')
    
    # Row 2: Magnitude spectrums
    im1 = axes[1, 0].imshow(spectrum1, cmap='hot')
    axes[1, 0].set_title('Magnitude Spectrum 1')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(spectrum2, cmap='hot')
    axes[1, 1].set_title('Magnitude Spectrum 2')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(spectrum3, cmap='hot')
    axes[1, 2].set_title('Magnitude Spectrum 3')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('frequency_domain_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved frequency domain visualization as 'frequency_domain_analysis.png'")


def show_frequency_explanation():
    """
    Explains what we see in the frequency domain
    """
    print("\n" + "="*80)
    print("FREQUENCY DOMAIN EXPLANATION")
    print("="*80)
    print("""
WHAT IS THE MAGNITUDE SPECTRUM?
- Shows the strength of different frequencies in the image
- Bright areas = strong frequencies (clear patterns)
- Dark areas = weak frequencies (subtle patterns)  
- Center = low frequencies (overall shape)
- Edges = high frequencies (fine details)

WHY THIS WORKS FOR PALMPRINTS:
- Palm lines create unique frequency patterns
- Different frequencies capture different aspects of palm texture
- The frequency distribution is unique to each person

COLOR INTERPRETATION:
- Hot colors (red/yellow) = high magnitude
- Cool colors (blue/black) = low magnitude
- The pattern is unique to each person
    """)


if __name__ == "__main__":
    print("PALMPRINT FREQUENCY DOMAIN VISUALIZATION")
    print("=" * 50)
    
    # Show explanation first
    show_frequency_explanation()
    
    # Create visualizations
    print("\nGenerating frequency domain visualizations...")
    visualize_frequency_domain()
    
    print("\nFrequency domain visualization complete!")
    print("Generated file:")
    print("   - frequency_domain_analysis.png")
