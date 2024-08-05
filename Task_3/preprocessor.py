import numpy as np
import PIL.Image as Image
import scipy.ndimage.interpolation as inter
import cv2

class ImageSkewCorrector:
    """
    Corrects skews of an image.

    Attributes:
        import_image: an PNG image. 
    """
    def __init__(self, import_image):
        """
        Initializes an image that will be corrected.

        Args:
        import_image: an PNG image. 
        """
        self.import_image = np.array(import_image)

    def find_score(self, angle):
        """
        Finds the best score for the skew correction.

        Args: 
            angle: an angle for the correction.

        Returns:
            score: a score for the correction.
        """
        data = inter.rotate(self.import_image, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score
    
    def correct_skew(self):
        """
        Applies the best score and saving image with the correct skew.

        Returns:
            img_skew_corrected: an image with a corrected skew
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
    Binarizes an image after skew correction.

    Attributes:
        img_skew_corrected: an PNG image with corrected skew. 
    """
    def __init__(self, img_skew_corrected):
        """
        Initializes an image.

        Args: 
            img_skew_corrected: an PNG image with corrected skew.
        """
        self.import_image = np.array(img_skew_corrected)

    def gray_conversion(self):
        """
        Converts a colored image to gray scale.

        Returns:
            gray_image: an image in gray scale.
        """
        self.gray_image = cv2.cvtColor(self.import_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def binarized_conversion (self, gray_image):
        """
        Converts a gray scale image to binarized image.

        Args: 
            gray_image: an image in gray scale.

        Returns:
            binarized_image: an image binarized.
        """
        binarized_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
        return binarized_image

class NoiseRemoval:
    """
    Removes noise.

    Attributes:
        binarized_image: an image after binarization. 
    """
    def __init__(self, binarized_image):
        """
        Initializes an image.

        Args: 
            binarized_imaged: an image after binarization. 
        """
        self.import_image = binarized_image

    def gaussian_blurring (self):
        """
        Uses Gaussian blurring for noise removal.

        Returns:
            noise_removed_image: an image after noise removal. 
        """
        noise_removed_image = cv2.GaussianBlur(self.import_image, (5,5),0)
        return noise_removed_image