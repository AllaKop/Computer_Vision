# File reads .pdf files and converts them as images png

import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from data_preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval

class PdfToImageConvertor:

# initializing import an image
    def __init__(self, input_file):
        self.input_file = input_file

# converting pdf file to an image png file
    def pdf_to_images(self, output_folder):
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

    # initializing import an image
    def __init__(self, image_path):
        self.image_path = image_path

    # preprocessing and saving an image
    def preprocess_image(self) :
        image = Image.open(self.image_path)

        # Converting image to numpy array for processing
        image_np = np.array(image)

        # Correct Skew
        skew_corrector = ImageSkewCorrector(image_np)
        corrected_image = skew_corrector.correct_skew()

        # Convert to Gray and Binarize
        binarizer = Binarization(corrected_image)
        gray_image = binarizer.gray_conversion()
        binarized_image = binarizer.binarized_conversion(gray_image)

        # Remove Noise
        noise_removal = NoiseRemoval(binarized_image)
        final_image = noise_removal.gaussian_blurring()

        return Image.fromarray(final_image)