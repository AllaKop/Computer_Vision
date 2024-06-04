# Trying the library on the test image

import easyocr 

reader = easyocr.Reader(['en','en']) # this needs to run only once to load the model into memory
result = reader.readtext('/Users/allakopiichenko/Desktop/test_image_1.jpeg', detail = 0, paragraph=True)

print(result)