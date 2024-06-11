# File reads .pdf files and converts them as images png

import os
import PyPDF2
from PIL import Image
from pdf2image import convert_from_path

class PdfToImageConvertor:

# initializing import image
    def __init__(self, input_file):
        self.input_file = input_file

    def pdf_to_images(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = convert_from_path(self.input_file)
        for i, image in enumerate(images):
            image.save(f'{output_folder}/page_{i + 1}.png')

class ImagePreProcessor:
   
    # initializing output image
    def __init__(self, input_file):
        self.input_file = input_file

    # savinf output image
    def saving_processed_image(input_file, output_folder):
        
        def preprocess_image(self):
            from data_preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval

            # Correct Skew
            skew_corrector = ImageSkewCorrector(self.input_file)
            corrected_image = skew_corrector.correct_skew()

            # Convert to Gray and Binarize
            binarizer = Binarization(corrected_image)
            gray_image = binarizer.gray_conversion()
            binarized_image = binarizer.binarized_conversion(gray_image)

            # Remove Noise
            noise_removal = NoiseRemoval(binarized_image)
            final_image = noise_removal.gaussian_blurring(binarized_image)

            return preprocess_image