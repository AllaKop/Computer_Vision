import numpy as np
import PIL.Image as Image
import scipy.ndimage.interpolation as inter
import cv2

class ImageSkewCorrector:
    def __init__(self, import_image):
        self.import_image = np.array(import_image)

    def find_score(self, angle):
        data = inter.rotate(self.import_image, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return score
    
    def correct_skew(self):
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
    def __init__(self, img_skew_corrected):
        self.import_image = np.array(img_skew_corrected)

    def gray_conversion(self):
        self.gray_image = cv2.cvtColor(self.import_image, cv2.COLOR_RGB2GRAY)
        return self.gray_image

    def binarized_conversion(self, gray_image):
        binarized_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binarized_image

class NoiseRemoval:
    def __init__(self, binarized_image):
        self.import_image = binarized_image

    def gaussian_blurring(self):
        noise_removed_image = cv2.GaussianBlur(self.import_image, (5, 5), 0)
        return noise_removed_image
