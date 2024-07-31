import click
from implementation import Input, PreProcessor, Layout, Output_Saver

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')
def launch_program(pdf_name, path):
    """
    CLI function to process a PDF file by converting it to images, preprocessing the images,
    detecting layout, and saving the results.

    Args: 
        pdf_name: a name of pdf file.
        path: a path to pdf file.
    """
    # Step 1: Convert PDF to images
    pdf_to_image_convertor = Input(pdf_name, path)
    image_paths = pdf_to_image_convertor.process_pdf()

    # Step 2: Preprocess images
    preprocessor = PreProcessor(image_paths)
    preprocessed_images = preprocessor.preprocess_images()

    # Step 3: Detect layout
    layout_processor = Layout(preprocessed_images)
    results = layout_processor.layout_detection()  # Changed to `layout_detection`

    # Step 4: Save results
    output_saver = Output_Saver(results, path)
    output_saver.saver()

if __name__ == '__main__':
    launch_program()

