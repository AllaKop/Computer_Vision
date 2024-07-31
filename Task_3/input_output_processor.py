import os
from pdf2image import convert_from_path
from PIL import Image

class PdfToImageConvertor:
    """
    Reads .pdf files and converts them to images (PNG).

    Attributes:
        input_file: A .pdf files that will be converted.
    """
    def __init__(self, input_file):
        """
        Initializes the input PDF file. Handles if the file does not exist.

        Args:
            input_file: A .pdf files that will be converted.

        Raises:
        FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")
        self.input_file = input_file

    def pdf_to_images(self, output_folder):
        """
        Converts the PDF file to PNG images and saves them to the output folder.
        Creates an output_folder if needed.

        Args:
            output_folder: The folder where the converted PNG images will be saved to.

        Returns:
            image_paths: A list of paths to the saved images.
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
    Saves processed images.

    Attributes: 
        output_folder: a folder to save images.
    """
    def __init__(self, image_with_boxes_pil):
        """
        Initializes an final output folder.

        Args: 
            final_images: final images
        """
        self.final_images = image_with_boxes_pil

    def save_images(self,  final_output):
        """
        Saves the processed (for now after layout) images to the output folder overwriting existing.

        Args: 
            final_output: A folder to same final images.

        Returns:
            final_image_paths: A list of images paths.
        """

        if not os.path.exists(self.final_output):
            os.makedirs(self.final_output)
        
        final_image_paths = []
        final_images = convert_from_path(self.final_images)
        for i, final_image in enumerate(final_images):
            final_image_path = os.path.join(self.final_output, f'page_{i + 1}_layout.png')
            final_image.save(final_image_path)
            final_image_paths.append(final_image_path)

        return final_image_paths