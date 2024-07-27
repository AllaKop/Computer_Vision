# CLI for running the program

import click
import os
from implementation import ImagePreProcessor, Layout_processor


@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')

def launch_program(pdf_name, path):
    pdf_paths = os.path.abspath(pdf_name)


    for pdf_path in pdf_paths:
        preprocessor = ImagePreProcessor(pdf_path)
        processed_image = preprocessor.preprocess_image()
        layouted_image = Layout_processor.define_layout()
    

if __name__ == '__main__':
    launch_program()