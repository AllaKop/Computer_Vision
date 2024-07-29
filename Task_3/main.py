import click
import os
from implementation import PdfProcessingPipeline

@click.command()
@click.argument('pdf_name')
@click.option('--image-folder', '-i', default='images', help='Folder to save images extracted from PDF')
@click.option('--result-folder', '-r', default='results', help='Folder to save processed results')
def launch_program(pdf_name, image_folder, result_folder):
    pdf_path = os.path.abspath(pdf_name)

    if not os.path.isfile(pdf_path):
        print(f"The file {pdf_path} does not exist.")
        return

    pipeline = PdfProcessingPipeline(pdf_path, image_folder, result_folder)
    pipeline.process_pdf()

    print(f"Processing complete. Results saved to {result_folder}")

if __name__ == '__main__':
    launch_program()
