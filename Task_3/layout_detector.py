import layoutparser as lp
import numpy as np
import torch
from detectron2 import model_zoo

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, import_image):
        """
        The method is initializing the image that will be segmented and the model that will be used
        """
        self.import_image = import_image[..., ::-1]  
        self.model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)

    def detect_image_layout(self):
        """
        The method is detecting the layout of the image
        """
        layout = self.model.detect(self.import_image)
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
        
        # Convert PIL image to NumPy array
        image_with_boxes_np = np.array(image_with_boxes_pil)
        return image_with_boxes_np, layout
