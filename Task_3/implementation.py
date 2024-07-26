import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from input_output_files_processor import PdfToImageConvertor

class ImagePreProcessor:
    """
    The class in bound to implement all classes
    """
    def __init__(self, image_path):
        """
        The method is initializing import an image
        """
        self.image_path = image_path

    def preprocess_image(self) :
        """
        The method is implementing Preprocessor class and saving image.  
        """
        image = Image.open(self.image_path)

        # Converting image to numpy array for processing
        image_np = np.array(image)

        # Correcting Skew
        skew_corrector = ImageSkewCorrector(image_np)
        corrected_image = skew_corrector.correct_skew()

        # Converting to Gray and Binarize
        binarizer = Binarization(corrected_image)
        gray_image = binarizer.gray_conversion()
        binarized_image = binarizer.binarized_conversion(gray_image)

        # Removing Noise
        noise_removal = NoiseRemoval(binarized_image)
        final_image = noise_removal.gaussian_blurring()

        return Image.fromarray(final_image)