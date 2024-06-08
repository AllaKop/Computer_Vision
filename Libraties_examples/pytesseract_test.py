# Trying the library on the test image

from PIL import Image

import pytesseract

image = '/Users/allakopiichenko/Desktop/test_image_1.png'

print(pytesseract.image_to_string(Image.open(image)))