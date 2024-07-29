import numpy as np
from PIL import Image
from layout_detector import Layout
from input_output_processor import ResultSaver

class Input:
    def __init__(self, pdf_path, output_folder):
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def process_pdf(self):
        from input_output_processor import PdfToImageConvertor  
        pdf_to_image_convertor = PdfToImageConvertor(self.pdf_path)
        return pdf_to_image_convertor.pdf_to_images(self.output_folder)

class PreProcessor:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def preprocess_images(self):
        from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval  
        processed_images = []
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            skew_corrector = ImageSkewCorrector(image_np)
            corrected_image = skew_corrector.correct_skew()

            binarizer = Binarization(corrected_image)
            gray_image = binarizer.gray_conversion()
            binarized_image = binarizer.binarized_conversion(gray_image)

            noise_removal = NoiseRemoval(binarized_image)
            preprocessed_image = noise_removal.gaussian_blurring()

            if preprocessed_image.dtype == np.float32 or preprocessed_image.dtype == np.float64:
                preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
            elif preprocessed_image.dtype != np.uint8:
                raise ValueError(f"Unexpected image dtype: {preprocessed_image.dtype}")

            processed_images.append(preprocessed_image)
        return processed_images


class Output_Saver:
    def __init__(self, processed_images, output_folder):
        self.output = processed_images
        self.output_folder = output_folder

    def saver(self):
        result_saver = ResultSaver(self.output_folder)
        result_saver.save_images(self.output)
