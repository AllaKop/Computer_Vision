import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import cv2
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout

class ImagePreProcessor:
    """
    The class in bound to implement all classes from preprocessor file
    """
    def __init__(self, image_path):
        """
        The method is initializing import an image
        """
        self.image_path = image_path

    def preprocess_image(self):
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
        preprocessed_image = noise_removal.gaussian_blurring()
        return preprocessed_image

class Layout_processor:
    """
    The class in bound to implement all classes from layout_detector file
    """
    
    def __init__(self, preprocessed_image):
        """
        The method is initializing import an image
        """
        self.import_image = preprocessed_image

    def define_layout(self):
        """
        Detects the layout of the preprocessed image and saves the result.
        """
        if self.import_image is None:
            raise RuntimeError("Preprocessed image is not available. Please call preprocess_image first.")

        # Convert numpy array back to PIL Image if necessary
        preprocessed_image_pil = Image.fromarray(self.import_image)

        # Initialize layout detector with the preprocessed image
        layout_detector = Layout(preprocessed_image_pil)
        image_with_boxes, layout = layout_detector.detect_image_layout()
