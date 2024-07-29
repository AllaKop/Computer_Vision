import layoutparser as lp
import numpy as np
from PIL import Image

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, noise_removed_image):
        """
        The method is initializing the image that will be segmented and the model that will be used
        """
        self.import_image = np.array(noise_removed_image)[..., ::-1]  # Convert to BGR if needed
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure'}
        )

    def detect_image_layout(self):
        """
        The method detects the layout of the image.
        """
        # Ensure the image is in the expected format (H, W, C) and values are in range [0, 255]
        if self.import_image.dtype != np.uint8:
            self.import_image = (self.import_image * 255).astype(np.uint8)

        layout = self.model.detect(self.import_image)
        
        # Ensure layout detection results are in expected format
        if not isinstance(layout, list):  # or check for other types as needed
            raise ValueError(f"Expected a list of detected objects but got {type(layout)}")

        # Draw boxes on the image
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
        image_with_boxes_np = np.array(image_with_boxes_pil)

        return image_with_boxes_np, layout
