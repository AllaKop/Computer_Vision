import os
from pdf2image import convert_from_path
from PIL import Image

class PdfToImageConvertor:
    """
    The class reads .pdf files and converts them to images (PNG).
    """
    def __init__(self, input_file):
        """
        Initializes the input PDF file path.
        """
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")
        self.input_file = input_file

    def pdf_to_images(self, output_folder):
        """
        Converts the PDF file to PNG images and saves them to the output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = convert_from_path(self.input_file)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f'page_{i + 1}.png')
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths

class ResultSaver:
    """
    The class saves processed images.
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def save_images(self, results):
        """
        Saves the processed images to the output folder.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for i, (image_with_boxes, layout) in enumerate(results):
            output_image_path = os.path.join(self.output_folder, f'page_{i + 1}_layout.png')
            Image.fromarray(image_with_boxes).save(output_image_path)
