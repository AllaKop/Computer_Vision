# Trying the library on the test image

import easyocr

image = '/Users/allakopiichenko/Desktop/test_image_1.png'

reader = easyocr.Reader(['en','en']) # this needs to run only once to load the model into memory
result = reader.readtext(image, decoder='wordbeamsearch', detail = 0, paragraph=True)

print(result)