import layoutparser as lp
import numpy as np
from PIL import Image

class Layout:
    """
    Uses preprocessed documents and creates layout of the document.

    Attributes: An input image (after preprocessing).
    """
    def __init__(self, noise_removed_image):
        """
        Initializes an image that will be segmented and the model that will be used.

        Args:
            noise_removed_image: an image after the last step (noise removing) of preprocessing.
        """
        self.import_image = np.array(noise_removed_image)[..., ::-1]  # Convert to BGR if needed
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure'}
        )

    def detect_image_layout(self):
        """
        Detects the layout of the image.

        Returns: 
            image_with_boxes_np: An image with created layout in NumPy array format.
            layout: a list of detected layout objects, bounding boxes, labels, confidence scores.
        """
        layout = self.model.detect(self.import_image)

        # Draw boxes on the image
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
        image_with_boxes_np = np.array(image_with_boxes_pil)

        return image_with_boxes_np, layout
