import layoutparser as lp
import cv2

class Segmentator:
    """
    The class uses preprocessed documents and creates layout of the document
    """
    def __init__(self, image):
        """
        The method is initializing image that will be segmented
        """
        self.image = image
        image = cv2.imread("/Users/allakopiichenko/Desktop/CV_Internship_Meduzzen/Task_2/output_images/processed_page_1.png")
        image = image[..., ::-1]

    def detect_image_layout(image):
        """
        The method is detecting layout of the image
        """
        model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        layout = model.detect(image)
        lp.draw_box(image, layout, box_width=3)
        type(layout) 


    

