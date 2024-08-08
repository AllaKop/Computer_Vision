import click
import os
from docx import Document

from input_output_processor import PdfToImageConvertor, ImagePreProcessor, Layout, Doc

"""
A CLI for running the program.
"""

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')

def launch_program(pdf_name, path):
    """
    Runs the code.

    Args:
        pdf_name (str): a name of pdf file
        path (str): a path to a pdf file
    """
    pdf_path = os.path.abspath(pdf_name)
    converter = PdfToImageConvertor(pdf_path)
    image_paths = converter.pdf_to_images(path)

    for image_path in image_paths:
        preprocessor = ImagePreProcessor(image_path)
        processed_image = preprocessor.preprocess_image()
        layout = Layout(processed_image)
        extracted_text = layout.layout_detection()
        document = Doc(extracted_text)
        document.creating_doc()

if __name__ == '__main__':
    launch_program()