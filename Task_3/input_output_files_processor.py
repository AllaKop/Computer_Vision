# File reads .pdf files and converts them as images png

import os
import PyPDF2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

class PdfToImageConvertor:
    """
    The class is bount to converting pdf files to png images
    """
    def __init__(self, input_file):
        """
        The method is initializing import an image
        """
        self.input_file = input_file

    def pdf_to_images(self, output_folder):
        """
        The method is converting pdf file to an image png file
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