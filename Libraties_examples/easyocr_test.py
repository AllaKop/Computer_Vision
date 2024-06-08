# Trying the library on the test image

import easyocr

image = '/Users/allakopiichenko/Desktop/test_image_1.png'
image_handwritten = '/Users/allakopiichenko/Desktop/test_image_2.png'

reader = easyocr.Reader(['en','en']) 

result = reader.readtext(image_handwritten, decoder='wordbeamsearch', detail = 0, paragraph=True)

print(result)

