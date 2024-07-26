# CLI for running the program

import click
import os
from input_output_files_preprocessor import PdfToImageConvertor, ImagePreProcessor

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')

def launch_program(pdf_name, path):
    pdf_path = os.path.abspath(pdf_name)
    converter = PdfToImageConvertor(pdf_path)
    image_paths = converter.pdf_to_images(path)

    for image_path in image_paths:
        preprocessor = ImagePreProcessor(image_path)
        processed_image = preprocessor.preprocess_image()
        processed_image.save(os.path.join(path, f"processed_{os.path.basename(image_path)}"))
    

if __name__ == '__main__':
    launch_program()