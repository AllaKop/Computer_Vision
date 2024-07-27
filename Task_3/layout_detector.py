import layoutparser as lp
import cv2
import os
import numpy as np

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, image_path):
        """
        The method is initializing image that will be segmented and model that will be used
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file at {image_path} does not exist.")
        
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Could not read the image at {image_path}. Check the image or/and image path integrity")

        self.image = self.image[..., ::-1]  

        try:
            self.model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
        except AttributeError:
            raise RuntimeError("Error initializing Detectron2LayoutModel. Check if the detectron2 and Pillow libraries are correctly installed and compatible.")

    def detect_image_layout(self):
        """
        The method is detecting layout of the image
        """
        layout = self.model.detect(self.image)
        image_with_boxes_pil = lp.draw_box(self.image, layout, box_width=3)
        
        # Convert PIL image to NumPy array
        image_with_boxes_np = np.array(image_with_boxes_pil)
        return image_with_boxes_np, layout       

