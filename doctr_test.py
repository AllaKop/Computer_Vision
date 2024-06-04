# Was not able to use libraries. Asks to many dependencies for MacOS

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

# Image
doc = DocumentFile.from_images("/Users/allakopiichenko/Desktop/test_image_1")

# Analyze
result = model(doc)