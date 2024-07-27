import os
from pdf2image import convert_from_path

class PdfToImageConvertor:
    """
    The class reads .pdf files and converts them as images (PNG)
    """
    def __init__(self, input_file):
        """
        The method initializes the input PDF file path.
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

        try:
            images = convert_from_path(self.input_file)
        except Exception as e:
            raise RuntimeError(f"An error occurred while converting PDF to images: {e}")

        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f'page_{i + 1}.png')
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths
