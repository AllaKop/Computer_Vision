import layoutparser as lp
import numpy as np
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

import preprocessor


class LayoutDetector:
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
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.9],
            label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
        )

    def process_image_layout(self) -> dict:
        """
        Process an image with detected layout.
        Extract text.

        Args:
            layout: a defined layout.

        Returns:
            dict: a dictionary containing the processed text for each block type. 

        """
        layout = self.model.detect(self.import_image)

        # Identify and categorize of different layout elements.
        blocks_dict = {block_type: lp.Layout([b for b in layout if b.type == block_type])
                       for block_type in ["Text", "Title", "List", "Table", "Figure"]}

        h, w = self.import_image.shape[:2]
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(self.import_image)

        for block_type, blocks in blocks_dict.items():
            left_blocks = blocks.filter_by(left_interval, center=True)
            left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
            right_blocks = lp.Layout([b for b in blocks if b not in left_blocks])
            right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
            blocks_dict[block_type] = lp.Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])

        ocr_agent = lp.TesseractAgent(languages='eng')
        for block_type, blocks in blocks_dict.items():
            for block in blocks:
                segment_image = (block.pad(left=5, right=5, top=5, bottom=5)
                                 .crop_image(self.import_image))
                text = ocr_agent.detect(segment_image)
                block.set(text=text, inplace=True)

        return blocks_dict
