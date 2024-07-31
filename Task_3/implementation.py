"""The file is bound to implement all other files and clases"""

import numpy as np
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout_detector
from input_output_processor import PdfToImageConvertor, ResultSaver


class Input:
    """
    Implements input processing - excepting .pdf file and converts it to image PNG (input_output_processor.py)

    Attributes:
        pdf_path: A path to a .pdf file.
        output_folder: A folder to save PNG files.
    """
    def __init__(self, pdf_path, output_folder):
        """
        Initializes pdf_path and output_folder.

        Args:
            pdf_path: A path to a .pdf file.
            output_folder: A folder to save PNG files. 
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def process_pdf(self):
        """
        An implementation of PdftoImageConvertot.

        Returns: 
        A folder with converted images.
        """
        pdf_to_image_convertor = PdfToImageConvertor(self.pdf_path)
        return pdf_to_image_convertor.pdf_to_images(self.output_folder)

class PreProcessor:
    """
    Implements preprocessing (preprocessor.py)

    Attributes:
        image_paths: A list of path of the images.
    """
    def __init__(self, image_paths):
        """
        Initializes image paths.

        Args: 
            image_path: a list of image paths.
        """
        self.image_paths = image_paths

    def preprocess_images(self):
        """
        Preprocessing images.

        Returns:
            processed_images: a list of paths to processed images.
        """
        processed_images = []
        for image_path in self.image_paths:
            image = Image.open(image_path)

            skew_corrector = ImageSkewCorrector(image)
            corrected_image = skew_corrector.correct_skew()

            binarizer = Binarization(corrected_image)
            gray_image = binarizer.gray_conversion()
            binarized_image = binarizer.binarized_conversion(gray_image)

            noise_removal = NoiseRemoval(binarized_image)
            preprocessed_image = noise_removal.gaussian_blurring()
            processed_images.append(preprocessed_image)
        return processed_images

class Layout:
     """
    Implements layout detection (layout_detector.py)

    Attributes:
        image_paths: A list of path of the images.
    """
def __init__(self, processed_images):
        """
        Initializes image paths.

        Args: 
            image_path: a list of image paths after preprocessing.
        """
        image_paths = processed_images
        self.preprocessed_image = Image.open(image_paths)

def layout_detection(self):
    """
    Detects layout.
    """
    layout_detector = Layout_detector(self.preprocessed_image)
    results = layout_detector.detect_image_layout()

    for i, (image_with_boxes_np, layout) in enumerate(results):
        output_image_path = (f'path/to/output_page_{i + 1}_layout.png')
        Image.fromarray(image_with_boxes_np).save(output_image_path)


class Output_Saver:
    """
    Implements saving of processed images (for now after layout) (input_output_processor.py)

    Attributes:
        output_image_path:  a list of paths to processed images.
        output_folder: an output folder.

    """
    def __init__(self, output_image_path, output_folder):
        """
        Initializes image paths.

        Args: 
            output_image_path:  a list of paths to images with layout.
            output_folder: an output folder.
        """
        self.output = output_image_path
        self.output_folder = output_folder

    def saver(self):
        """
        Saves images to output_folder
        """
        result_saver = ResultSaver(self.output_folder)
        result_saver.save_images(self.output)
