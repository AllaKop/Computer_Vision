# implementation.py
import numpy as np
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval
from layout_detector import Layout
from input_output_processor import PdfToImageConvertor, ResultSaver

class Input:
    """
    The class implements input processing
    """
    def __init__(self, pdf_path, output_folder):
        """
        Initializes the class with the PDF path and the output folder.
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def process_pdf(self):
        """
        Processes the PDF file by converting it to images.
        """
        pdf_to_image_convertor = PdfToImageConvertor(self.pdf_path)
        return pdf_to_image_convertor.pdf_to_images(self.output_folder)

class PreProcessor:
    """
    The class implements preprocessing
    """
    def __init__(self, image_paths):
        """
        Initializes the class with paths of images converted from the pdf file.
        """
        self.image_paths = image_paths

    def preprocess_images(self):
        """
        Processes each image in the list of image paths, and returns the processed images.
        """
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

class LayoutProcessor:
    """
    The class implements layout 
    """
    def __init__(self, preprocessed_images):
        """
        Initializes the class with preprocessed images.
        """
        self.preprocessed_images = preprocessed_images
        self.model = Layout('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

    def define_layout(self):
        """
        Detects the layout of the preprocessed images and returns results.
        """
        results = []
        for preprocessed_image in self.preprocessed_images:
            if preprocessed_image is None:
                raise RuntimeError("Preprocessed image is not available.")

            preprocessed_image_pil = Image.fromarray(preprocessed_image)
            layout = self.model.detect(np.array(preprocessed_image_pil))
            image_with_boxes_pil = self.model.draw_box(preprocessed_image_pil, layout, box_width=3)
            image_with_boxes_np = np.array(image_with_boxes_pil)

            results.append((image_with_boxes_np, layout))
        return results

class Output_Saver:
    """
    The class implements output processing
    """
    def __init__(self, results, output_folder):
        """
        Initializes the class with layout images.
        """
        self.output = results
        self.output_folder = output_folder

    def saver(self):
        """
        Saves results
        """
        result_saver = ResultSaver(self.output_folder)
        result_saver.save_images(self.output)
