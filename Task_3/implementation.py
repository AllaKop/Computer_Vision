import numpy as np
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout
from input_output_processor import PdfToImageConvertor, ResultSaver

class Input:
    def __init__(self, pdf_path, output_folder):
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def process_pdf(self):
        pdf_to_image_convertor = PdfToImageConvertor(self.pdf_path)
        return pdf_to_image_convertor.pdf_to_images(self.output_folder)

class PreProcessor:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def preprocess_images(self):
        processed_images = []
        for image_path in self.image_paths:
            image = Image.open(image_path)
            image_np = np.array(image)

            # Correct Skew
            skew_corrector = ImageSkewCorrector(image_np)
            corrected_image = skew_corrector.correct_skew()

            # Convert to Gray and Binarize
            binarizer = Binarization(corrected_image)
            gray_image = binarizer.gray_conversion()
            binarized_image = binarizer.binarized_conversion(gray_image)

            # Remove Noise
            noise_removal = NoiseRemoval(binarized_image)
            preprocessed_image = noise_removal.gaussian_blurring()

            processed_images.append(preprocessed_image)
        return processed_images

class Output_Saver:
    def __init__(self, processed_images, output_folder):
        self.output = processed_images
        self.output_folder = output_folder

    def saver(self):
        result_saver = ResultSaver(self.output_folder)
        result_saver.save_images(self.output)
