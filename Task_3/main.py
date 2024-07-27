import click
import os
from implementation import ImagePreProcessor, LayoutProcessor
from PIL import Image

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')
def launch_program(pdf_name, path):
    pdf_path = os.path.abspath(pdf_name)
    output_folder = os.path.abspath(path)

    if not os.path.isfile(pdf_path):
        print(f"The file {pdf_path} does not exist.")
        return

    # Initialize ImagePreProcessor
    preprocessor = ImagePreProcessor(pdf_path, output_folder)
    processed_images = preprocessor.preprocess_image()

    if not processed_images:
        print("No images were processed.")
        return

    # Initialize LayoutProcessor
    layout_processor = LayoutProcessor(processed_images)
    results = layout_processor.define_layout()

    # Save output images
    for i, (image_with_boxes, _) in enumerate(results):
        output_image_path = os.path.join(output_folder, f'layout_detected_page_{i + 1}.png')
        Image.fromarray(image_with_boxes).save(output_image_path)
        print(f"Layout detected image for page {i + 1} saved to {output_image_path}")

if __name__ == '__main__':
    launch_program()
