import numpy as np
from PIL import Image
from preprocessor import ImageSkewCorrector, Binarization, NoiseRemoval

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
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
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

            # Ensure preprocessed image is in correct numerical format
            if preprocessed_image.dtype == np.float32 or preprocessed_image.dtype == np.float64:
                preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
            elif preprocessed_image.dtype != np.uint8:
                raise ValueError(f"Unexpected image dtype: {preprocessed_image.dtype}")

            processed_images.append(preprocessed_image)
        return processed_images
