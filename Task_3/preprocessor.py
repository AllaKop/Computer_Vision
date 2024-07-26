import numpy as np
import PIL.Image as Image
import scipy.ndimage.interpolation as inter
import cv2

class ImageSkewCorrector:
    """
    The class is bound to correct skews of the images
    """
    def __init__(self, import_image):
        """
        The method is initializing import image
        """
        self.import_image = np.array(import_image)

    def find_score(self, angle):
        """
        The method is finding the best score for the skew correction
        """
        data = inter.rotate(self.import_image, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score
    
    def correct_skew(self):
        """
        The method is applying the best score and saving image with the correct skew
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
    The class is bound to convert colored images firstly to gray scale colours and then to binarized image
    """
    def __init__(self, img_skew_corrected):
        """
        The method is initializing import image
        """
        self.import_image = np.array(img_skew_corrected)

    def gray_conversion(self):
        """
        The method is converting colored image to gray scale
        """
        self.gray_image = cv2.cvtColor(self.import_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def binarized_conversion (self, gray_image):
        """
        The method is converting gray scale image to binarized image
        """
        binarized_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
        return binarized_image

class NoiseRemoval:
    """
    The class is bound to remove noises
    """
    def __init__(self, binarized_image):
        """
        The method is initializing import image
        """
        self.import_image = binarized_image

    def gaussian_blurring (self):
        """
        The method is using Gaussian blurring for noise removal
        """
        noise_removed_image = cv2.GaussianBlur(self.import_image, (5,5),0)
        return noise_removed_image
