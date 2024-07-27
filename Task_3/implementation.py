import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import cv2
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from input_output_files_processor import PdfToImageConvertor
from layout_detector import Layout

class ImagePreProcessor:
    """
    The class in bound to implement all classes
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
        final_image = noise_removal.gaussian_blurring()

        return Image.fromarray(final_image)

    def define_layout(self):
        """
        This method is implementing Layout class 
        """
        layout_detector = Layout(PdfToImageConvertor.image_paths)
        image_with_boxes, layout = layout_detector.detect_image_layout()

        # Save the resulting image
        output_path = "/Users/allakopiichenko/Desktop/CV_Internship_Meduzzen/Task_3/processed_page_1_with_boxes.png"
        cv2.imwrite(output_path, image_with_boxes[..., ::-1])  # Convert RGB back to BGR for saving
        print(f"Image with layout boxes saved at {output_path}")
      
