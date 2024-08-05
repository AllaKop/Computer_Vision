import layoutparser as lp
import numpy as np
import preprocessor
from PIL import Image

class Layout_detector:
    """
    Uses preprocessed documents and creates layout of the document.

    Attributes: an input image (after preprocessing).
    """
    def __init__(self, noise_removed_image):
        """
        Initializes an image that will be segmented and the model that will be used.

        Args:
            noise_removed_image: an image after the last step (noise removing) of preprocessing.
        """
        noise_removed_image = Image.fromarray(noise_removed_image)
        noise_removed_image = noise_removed_image.convert('RGB')
        self.import_image = np.array(noise_removed_image)[..., ::-1] 
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure'}
        )

    def detect_image_layout(self):
        """
        Detects the layout of the image.

        Returns: 
            images_with_layout: a list of images with layouts paths.
        """
        layout = self.model.detect(self.import_image)

        # Draw boxes on the image
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)

        return image_with_boxes_pil
