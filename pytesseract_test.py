# Trying the library on the test image

from PIL import Image

import pytesseract

print(pytesseract.image_to_string(Image.open('/Users/allakopiichenko/Desktop/test_image_1.jpeg')))