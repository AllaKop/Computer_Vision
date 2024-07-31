import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout_detector
from input_output_processor import PdfToImageConvertor, ResultSaver

class Implementation:
    """
    Implements:
        - input_output_processor.py
        - preprocessor.py
        - layout_detector.py

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
            final_output: A folder to save final images.
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def process_pdf(self):
        """
        Converts PDF to images using PdfToImageConvertor.

        Returns: 
            images_paths: A list of paths to the converted images.
        """
        pdf_to_image_convertor = PdfToImageConvertor(self.pdf_path)
        images_paths = pdf_to_image_convertor.pdf_to_images(self.output_folder)
        return images_paths

    def preprocess_images(self, images_paths):
        """
        Applies preprocessing steps to images.

        Args:
            images_paths: A list of image paths.

        Returns: 
            preprocessed_images: A list of processed images.
        """
        preprocessed_images = []
        for image_path in images_paths:
            image = Image.open(image_path)

            skew_corrector = ImageSkewCorrector(image)
            corrected_image = skew_corrector.correct_skew()

            binarizer = Binarization(corrected_image)
            gray_image = binarizer.gray_conversion()
            binarized_image = binarizer.binarized_conversion(gray_image)

            noise_removal = NoiseRemoval(binarized_image)
            preprocessed_image = noise_removal.gaussian_blurring()
            preprocessed_images.append(preprocessed_image)
        return preprocessed_images

    def layout_detection(self, preprocessed_images):
        """
        Detects layout in preprocessed images.

        Args:
            preprocessed_images: A list of preprocessed images.

        Returns: 
            layout_images_paths: A list of paths to images with detected layouts.
        """
        layout_images_paths = []
        for image in preprocessed_images:
            layout_detector = Layout_detector(image)
            layout_images_paths.extend(layout_detector.detect_image_layout())

        return layout_images_paths

    def saver(self, layout_images_paths):
        """
        Saves images with detected layouts to the output folder.

        Args:
            layout_images_paths: A list of paths to images with detected layouts.

        Returns:
            None
        """
        result_saver = ResultSaver(self.output_folder)
        result_saver.save_images(layout_images_paths)
