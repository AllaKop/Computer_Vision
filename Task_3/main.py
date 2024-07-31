import click
from implementation import Implementation

@click.command()
@click.argument('pdf_name')
@click.option('--path', '-p', default='output_images', help='Output folder path')

def launch_program(pdf_name, path):
    """
    CLI function to process a PDF file by converting it to images, preprocessing the images,
    detecting layout, and saving the results.

    Args: 
        pdf_name: A name of pdf file.
        path: A path to a pdf file.
    """
    # Create an instance of the Implementation class
    impl = Implementation(pdf_name, path)
    
    # Step 1: Convert PDF to images
    image_paths = impl.process_pdf()

    # Step 2: Preprocess images
    preprocessed_images = impl.preprocess_images(image_paths)

    # Step 3: Detect layout
    layout_images_paths = impl.layout_detection(preprocessed_images)

    # Step 4: Save results
    impl.saver(layout_images_paths)
    
if __name__ == '__main__':
    launch_program()
