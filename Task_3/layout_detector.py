import layoutparser as lp
import numpy as np

class Layout:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, preprocessed_images):
        """
        The method is initializing the images that will be segmented and the model that will be used
        """
        self.preprocessed_images = preprocessed_images
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure'}
        )

    def detect_image_layout(self):
        """
        The method detects the layout of the images.
        """
        results = []
        for image in self.preprocessed_images:
            # Ensure image is in correct numerical format
            if image.dtype != np.uint8:
                raise ValueError(f"Image dtype should be np.uint8, but got {image.dtype}")

            layout = self.model.detect(image)
            
            # Ensure layout detection results are in the expected format
            if not isinstance(layout, list):
                raise ValueError(f"Expected a list of detected objects but got {type(layout)}")

            # Draw boxes on the image
            image_with_boxes_pil = lp.draw_box(image, layout, box_width=3)
            image_with_boxes_np = np.array(image_with_boxes_pil)

            results.append((image_with_boxes_np, layout))
        return results
