import click
import os
from implementation import ImagePreProcessor, LayoutProcessor
from PIL import Image

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')
def launch_program(pdf_name, path):
    pdf_path = os.path.abspath(pdf_name)

    if not os.path.isfile(pdf_path):
        print(f"The file {pdf_path} does not exist.")
        return

    preprocessor = ImagePreProcessor(pdf_path)
    processed_image = preprocessor.preprocess_image()
    layout_processor = LayoutProcessor(processed_image)
    image_with_boxes, layout = layout_processor.define_layout()

    # Save the output if needed
    output_image_path = os.path.join(path, 'layout_detected_image.png')
    Image.fromarray(image_with_boxes).save(output_image_path)
    print(f"Layout detected image saved to {output_image_path}")

if __name__ == '__main__':
    launch_program()
