import layoutparser as lp
import numpy as np

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, import_image):
        """
        The method is initializing the image that will be segmented and the model that will be used
        """
        self.import_image = import_image[..., ::-1]
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure'}
        )

    def detect_image_layout(self):
        """
        The method is detecting the layout of the image
        """
        layout = self.model.detect(self.import_image)
        image_with_boxes_pil = lp.draw_box(self.import_image, layout, box_width=3)
        image_with_boxes_np = np.array(image_with_boxes_pil)
        return image_with_boxes_np, layout
