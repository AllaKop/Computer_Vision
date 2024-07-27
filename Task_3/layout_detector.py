import layoutparser as lp
import cv2
import os
import numpy as np
import detectron2

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document.
    """
    def __init__(self, import_image):
        """
        Initializes the image to be segmented and the model to be used.
        """
        self.import_image = import_image[..., ::-1]  # Convert BGR to RGB
        try:
            # Replace with the correct path to your model weights
            self.model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                weights='path_to_your_weights/model_final.pth'
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing Detectron2LayoutModel: {e}")

    def detect_image_layout(self):
        """
        Detects the layout of the image.
        """
        try:
            layout = self.model.detect(self.import_image)
            image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
            
            # Convert PIL image to NumPy array
            image_with_boxes_np = np.array(image_with_boxes_pil)
            return image_with_boxes_np, layout
        except Exception as e:
            raise RuntimeError(f"Error detecting image layout: {e}")
