import numpy as np
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout
from input_output_processor import PdfToImageConvertor

class ImagePreProcessor:
    def __init__(self, pdf_path, output_folder):
        """
        Initializes the class with PDF path.
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def preprocess_image(self):
        """
        Converts PDF to images, processes each image, and returns the processed images.
        """
        # Convert PDF to images
        convertor = PdfToImageConvertor(self.pdf_path)
        image_paths = convertor.pdf_to_images(self.output_folder)

        processed_images = []
        for image_path in image_paths:
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
    def __init__(self, preprocessed_images):
        """
        Initializes the class with preprocessed images.
        """
        self.preprocessed_images = preprocessed_images

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
