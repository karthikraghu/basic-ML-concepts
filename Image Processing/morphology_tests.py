import numpy as np
import cv2
import morphological
import unittest
from PIL import Image


def create_erosion_test_image():
    image = np.zeros((16, 16), dtype=np.uint8)
    cv2.circle(image, (5, 8), 4, 1, -1)  # type: ignore
    cv2.circle(image, (11, 8), 4, 1, -1)  # type: ignore
    return image


def create_dilation_test_image():
    image = np.zeros((16, 16), dtype=np.uint8)
    cv2.circle(image, (8, 8), 6, 1, -1)  # type: ignore
    cv2.circle(image, (8, 8), 3, 0, -1)  # type: ignore
    return image


def create_opening_test_image():
    image = np.zeros((16, 16), dtype=np.uint8)
    cv2.circle(image, (5, 8), 4, 1, -1)  # type: ignore
    image[12, 12] = 1
    return image


def create_closing_test_image():
    image = np.zeros((16, 16), dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (13, 13), 1, -1)
    cv2.rectangle(image, (6, 6), (9, 9), 0, -1)
    image[8, 8] = 1
    return image


def load_image(filepath):
    # Load and convert back to binary (0/1).
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    return (arr > 128).astype(np.uint8)


class TestExtractRegion(unittest.TestCase):
    def test_extract_region(self):
        img = np.arange(1, 26).reshape(5, 5)
        padded = morphological.pad_image(img, 1)

        region = morphological.extract_region(padded, 3, 3, 3)
        expected = np.array([[ 7,  8,  9],
                             [12, 13, 14],
                             [17, 18, 19]])
        np.testing.assert_array_equal(region, expected)

        region2 = morphological.extract_region(padded, 1, 1, 3)
        expected2 = np.array([[0, 0, 0],
                              [0, 1, 2],
                              [0, 6, 7]])
        np.testing.assert_array_equal(region2, expected2)

        region3 = morphological.extract_region(padded, 3, 3, 5)
        expected3 = img
        np.testing.assert_array_equal(region3, expected3)


class TestMorphologicalReferenceImages(unittest.TestCase):
    def setUp(self):
        # Structuring elements.
        self.ses = {
            3: np.ones((3, 3), dtype=np.uint8),
            5: np.ones((5, 5), dtype=np.uint8)
        }

    def test_erosion_images(self):
        img = create_erosion_test_image()
        for se_size, se in self.ses.items():
            expected = load_image(f"data/results/erosion_{se_size}x{se_size}.png")
            result = morphological.erode_binary(img, se)
            np.testing.assert_array_equal(result, expected,
                                          err_msg=f"Erosion failed for SE {se_size}x{se_size}")

    def test_dilation_images(self):
        img = create_dilation_test_image()
        for se_size, se in self.ses.items():
            expected = load_image(f"data/results/dilation_{se_size}x{se_size}.png")
            result = morphological.dilate_binary(img, se)
            np.testing.assert_array_equal(result, expected,
                                          err_msg=f"Dilation failed for SE {se_size}x{se_size}")

    def test_opening_images(self):
        img = create_opening_test_image()
        for se_size, se in self.ses.items():
            expected = load_image(f"data/results/opening_{se_size}x{se_size}.png")
            result = morphological.open_binary(img, se)
            np.testing.assert_array_equal(result, expected,
                                          err_msg=f"Opening failed for SE {se_size}x{se_size}")

    def test_closing_images(self):
        img = create_closing_test_image()
        for se_size, se in self.ses.items():
            expected = load_image(f"data/results/closing_{se_size}x{se_size}.png")
            result = morphological.close_binary(img, se)
            np.testing.assert_array_equal(result, expected,
                                          err_msg=f"Closing failed for SE {se_size}x{se_size}")


if __name__ == "__main__":
    unittest.main()
