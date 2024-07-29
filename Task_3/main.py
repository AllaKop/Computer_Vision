import click
from implementation import Input, PreProcessor, LayoutProcessor, Output_Saver

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')
def launch_program(pdf_name, path):
    """
    CLI function to process a PDF file by converting it to images, preprocessing the images,
    detecting layout, and saving the results.
    """
    # Step 1: Convert PDF to images
    pdf_to_image_convertor = Input(pdf_name, path)  # Pass the path as the output folder
    image_paths = pdf_to_image_convertor.process_pdf()

    # Step 2: Preprocess images
    preprocessor = PreProcessor(image_paths)
    preprocessed_images = preprocessor.preprocess_images()

    # Step 3: Detect layout
    layout_processor = LayoutProcessor(preprocessed_images)
    results = layout_processor.define_layout()

    # Step 4: Save results
    output_saver = Output_Saver(results, path)
    output_saver.saver()

if __name__ == '__main__':
    launch_program()
