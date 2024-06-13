# CLI for running the program

import click
from PIL import Image
import os
import cv2
from input_output_files_preprocessor import PdfToImageConvertor, ImagePreProcessor

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')

def launch_program(pdf_name, path):
    pdf_path = os.path.abspath(pdf_name)
    converter = PdfToImageConvertor(pdf_path)
    converter.pdf_to_images(path)

    for filename in os.listdir(path):
        if filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = Image.open(image_path)
            preprocessor = ImagePreProcessor(image)
            processed_image = preprocessor.preprocess_image()
            processed_image.save(os.path.join(path, f"processed_{filename}"))
    

if __name__ == '__main__':
    launch_program()