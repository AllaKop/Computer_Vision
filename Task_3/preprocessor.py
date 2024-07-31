import numpy as np
import PIL.Image as Image
import scipy.ndimage.interpolation as inter
import cv2

class ImageSkewCorrector:
    """
    Corrects skew of the input images.

    Attributes:
        import_image: image that will be skew corrected.
    """
    def __init__(self, import_image):
        """
        Initializes an import_image and converts in to NumPy array.

        Args: 
            import_image: an input image.
        """
        import_image = import_image.convert('RGB')
        self.import_image = np.array(import_image)

    def find_score(self, angle):
        """
        Finds the best score for the screw correction.

        Args: 
            angle: An angle for the correction.

        Returns:
            score: An computed score, which is used to evaluate how well the image aligns for the given rotation angle.
        """
        data = inter.rotate(self.import_image, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score
    
    def correct_skew(self):
        """
        Corrects skew of an image.

        Returns: 
            img_skew_corrected: an image with corrected skew.
        """
        delta = 1
        limit = 5
        angles = np.arange(-limit, limit + delta, delta)
        scores = [self.find_score(angle) for angle in angles]

        best_angle = angles[scores.index(max(scores))]
        print('Best angle:', best_angle)

        data = inter.rotate(self.import_image, best_angle, reshape=False, order=0)
        img_skew_corrected = Image.fromarray(data.astype('uint8')).convert('RGB')
        return img_skew_corrected

class Binarization:
    """
    Binatizes the skew corrected image RGB.

    Attributes:
        import_image: An input image
    """
    def __init__(self, img_skew_corrected):
        """
        Initializes an import image and assign it with the value of the RGB image after skew correction converted to Numpy array.

        Args:
            img_skew_corrected: An RGB image after skew correction
        """
        self.import_image = np.array(img_skew_corrected)

    def gray_conversion(self):
        """
        Converts an RGB image to gray shades image.

        Returns: 
            gray_image: An image after gray shades correction.
        """
        gray_image = cv2.cvtColor(self.import_image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def binarized_conversion(self, gray_image):
        """
        Binarizes a gray shades image.

        Returns:
            binarized_image: an image of binarized colors. 
        """
        binarized_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binarized_image

class NoiseRemoval:
    """
    Removes noises of a binarized image.

    Attributes:
        import_image: An input image.
    """
    def __init__(self, binarized_image):
        """
        Initializes an import_image and assigns it with value of binarized image.
        """
        self.import_image = binarized_image

    def gaussian_blurring(self):
        """
        Removes noises from an image.

        Returns:
            noise_removed_image: an image after removed noises.
        """
        noise_removed_image = cv2.GaussianBlur(self.import_image, (5, 5), 0)
        return noise_removed_image
