import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout_detector

class PdfToImageConvertor:
    """
    Converts a pdf file to an png image.

    Attributes:
        input_file: a pdf file. 
    """
    def __init__(self, input_file):
        """
        Initializes import of a pdf file.

        Args:
        input_file: a pdf file. 
        """
        self.input_file = input_file

    def pdf_to_images(self, output_folder):
        """
        Converts a pdf file to an image PNG.

        Args: 
            output_folder: a folder to save an image.

        Returns:
            image_paths: a list of an image path.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = convert_from_path(self.input_file)
        image_paths = []
        for i, image in enumerate(images):
            image_path = f'{output_folder}/page_{i + 1}.png'
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths

class ImagePreProcessor:
    """
    Implements preprocessor.py.

    Attributes:
        image_paths: a list of an image path. 
    """
    # initializing import an image
    def __init__(self, image_path):
        """
        Initializes an image_path.

        Args:
        image_paths: a list of an image path.  
        """
        self.image_path = image_path

    def preprocess_image(self) :
        """
        Preprocess and saves an image.

        Returns:
            final_image_array: an image (NumPy array format) after preprocessing.
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
        final_image_array = noise_removal.gaussian_blurring()
        return final_image_array

class Layout:
    """
    Implements layout_detector.py

    Attributes:
        image_array: an image (NumPy array format) after preprocessing. 
    """
    def __init__(self, image_array):
        """
        Initializes an image.

        Args:
            image_array: an image (NumPy array format) after preprocessing. 
        """
        self.image_array = image_array
    
    def layout_detection(self):
        """
        Detects layout in preprocessed images.

        Returns: 
            image_with_layout_array: an image with layout (NumPy array format)
        """
        layout_detector = Layout_detector(self.image_array)
        image_with_layout_array = layout_detector.detect_image_layout()
        return image_with_layout_array