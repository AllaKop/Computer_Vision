import layoutparser as lp
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, import_image):
        """
        The method is initializing the image that will be segmented and the model that will be used
        """
        self.import_image = import_image[..., ::-1]

        # Initialize Detectron2 config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # Set threshold
        self.model = DefaultPredictor(self.cfg)

    def detect_image_layout(self):
        """
        The method is detecting the layout of the image
        """
        outputs = self.model(self.import_image)
        instances = outputs["instances"].to("cpu")
        layout = lp.Layout(instances)  # Create a Layout object from Detectron2 instances
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
        
        # Convert PIL image to NumPy array
        image_with_boxes_np = np.array(image_with_boxes_pil)
        return image_with_boxes_np, layout
