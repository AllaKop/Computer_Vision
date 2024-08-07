import layoutparser as lp
import numpy as np
import preprocessor
from PIL import Image

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

    def process_image_layout(self):
        """
        Process an image with detected layout.
        Extract text.

        Args:
            layout: a defined layout.

        Returns:

        """
        layout = self.model.detect(self.import_image)

        # Identify and categorize of different layout elements.
        text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
        title_blocks = lp.Layout([b for b in layout if b.type=='Title'])
        list_blocks = lp.Layout([b for b in layout if b.type=='List'])
        table_blocks = lp.Layout([b for b in layout if b.type=='Table'])
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

        # Get image dimentions.
        h, w = self.import_image.shape[:2]

        # Define left interval.
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(self.import_image)

        # Filter and sort left layout elements.
        left_blocks_text = text_blocks.filter_by(left_interval, center=True)
        left_blocks_text.sort(key = lambda b:b.coordinates[1], inplace=True) # text elements

        left_blocks_title = title_blocks.filter_by(left_interval, center=True)
        left_blocks_title.sort(key = lambda b:b.coordinates[1], inplace=True) # title elements

        left_blocks_list = list_blocks.filter_by(left_interval, center=True)
        left_blocks_list.sort(key = lambda b:b.coordinates[1], inplace=True) # list elements

        left_blocks_table = table_blocks.filter_by(left_interval, center=True)
        left_blocks_table.sort(key = lambda b:b.coordinates[1], inplace=True) # table elements

        left_blocks_figure = figure_blocks.filter_by(left_interval, center=True)
        left_blocks_figure.sort(key = lambda b:b.coordinates[1], inplace=True) # figure elements

        # Filter and sort right layout elements.
        right_blocks_text = lp.layout([b for b in text_blocks if b not in left_blocks_text])
        right_blocks_text.sort(key = lambda b:b.coordinates[1], inplace=True) # text elements

        right_blocks_title = lp.layout([b for b in title_blocks if b not in left_blocks_title])
        right_blocks_title.sort(key = lambda b:b.coordinates[1], inplace=True) # title elements

        right_blocks_list = lp.layout([b for b in list_blocks if b not in left_blocks_list])
        right_blocks_list.sort(key = lambda b:b.coordinates[1], inplace=True) # list elements

        right_blocks_table = lp.layout([b for b in table_blocks if b not in left_blocks_table])
        right_blocks_table.sort(key = lambda b:b.coordinates[1], inplace=True) # table elements

        right_blocks_figure = lp.layout([b for b in figure_blocks if b not in left_blocks_figure])
        right_blocks_figure.sort(key = lambda b:b.coordinates[1], inplace=True) # figure elements

        # Combine and index blocks
        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks_text + right_blocks_text)])
        title_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks_title + right_blocks_title)])
        list_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks_list + right_blocks_list)])
        table_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks_table + right_blocks_table)])
        figure_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks_figure + right_blocks_figure)])
        
        # OCR process for all blocks.
        ocr_agent = lp.TesseractAgent(languages='eng')
        
        blocks_dict = {
            'text_blocks': text_blocks,
            'title_blocks': title_blocks,
            'list_blocks': list_blocks,
            'table_blocks': table_blocks,
            'figure_blocks': figure_blocks
        }
        
        for block_type, blocks in blocks_dict.items():
            for block in blocks:
                segment_image = (block
                                    .pad(left=5, right=5, top=5, bottom=5)
                                    .crop_image(self.import_image))
                text = ocr_agent.detect(segment_image)
                block.set(text=text, inplace=True)
        
        # Print detected texts for each block type.
        for block_type, blocks in blocks_dict.items():
            print(f"{block_type}:")
            for txt in blocks.get_texts():
                print(txt, end='\n---\n')
        
        return blocks_dict
