# File for data preprocessing before AI processing

import numpy as np
import PIL.Image as im
import scipy.ndimage.interpolation as inter
import cv2

class ImageSkewCorrector:

    # initializing import image
    def __init__(self, input_file):
        self.input_file = input_file

    # finding the best score for the skew correction
    def find_score(self, angle):
        data = inter.rotate(self.input_file, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score
    
    # applying the best score and saving image with the correct skew
    def correct_skew(self):
        delta = 1
        limit = 5
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = self.find_score(angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        print('Best angle: {}'.format(best_angle))

        data = inter.rotate(self.input_file, best_angle, reshape=False, order=0)
        img_skew_corrected = im.fromarray((data).astype('uint8')).convert('RGB')
        return img_skew_corrected

class Binarization:

    # initializing import image
    def __init__(self, img_skew_corrected):
        self.img_skew_corrected = img_skew_corrected

    # converting colored image to gray scale
    def gray_conversion(self, img_skew_corrected):
        gray_image = cv2.cvtColor(self.img_skew_corrected, cv2.COLOR_BGR2GRAY)
        return gray_image

    # converting gray scale image to binarized image
    def binarized_conversion (self, gray_image):
        binarized_image = cv2.adaptiveThreshold(self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
        return binarized_image

class NoiseRemoval:

    # initializing import image
    def __init__(self, binarized_image):
        self.binarized_image = binarized_image

    # using Gaussian blurring for noise removal
    def gaussian_blurring (self, binarized_image):
        noise_removed_image = cv2.GaussianBlur(self.binarized_image, (5,5),0)
        return noise_removed_image
