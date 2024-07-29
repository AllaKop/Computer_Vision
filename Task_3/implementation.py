import numpy as np
import os
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout
from input_output_processor import PdfToImageConvertor, ResultSaver

class ImagePreProcessor:
    def __init__(self, image_path):
        """
        Initializes the class with output_folder of images converted from pdf file.
        """
        self.image_path = image_path

    def preprocess_image(self):
        """
        Processes each image in the folder, and returns the processed images.
        """
        processed_images = []
        for image_name in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, image_name)
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

class LayoutProcessor:
    def __init__(self, processed_images):
        """
        Initializes the class with preprocessed images.
        """
        self.preprocessed_images = processed_images

    def define_layout(self):
        """
        Detects the layout of the preprocessed images and returns results.
        """
        results = []
        for preprocessed_image in self.preprocessed_images:
            if preprocessed_image is None:
                raise RuntimeError("Preprocessed image is not available.")

            preprocessed_image_pil = Image.fromarray(preprocessed_image)
            layout_detector = Layout(np.array(preprocessed_image_pil))
            image_with_boxes, layout = layout_detector.detect_image_layout()
            results.append((image_with_boxes, layout))
        return results


class PdfProcessingPipeline:
    def __init__(self, pdf_path, image_folder, result_folder):
        self.pdf_path = pdf_path
        self.image_folder = image_folder
        self.result_folder = result_folder

    def process_pdf(self):
        # Convert PDF to images
        convertor = PdfToImageConvertor(self.pdf_path)
        image_paths = convertor.pdf_to_images(self.image_folder)

        # Preprocess images
        preprocessor = ImagePreProcessor(image_paths)
        processed_images = preprocessor.preprocess_image()

        # Detect layout
        layout_processor = LayoutProcessor(processed_images)
        results = layout_processor.define_layout()

        # Save results
        saver = ResultSaver(self.result_folder)
        saver.save_results(results)