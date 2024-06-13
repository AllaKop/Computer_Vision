# Test image
image_path= '/Users/allakopiichenko/Desktop/test_image_1.png'
image_handwritten_path = '/Users/allakopiichenko/Desktop/test_image_2.png'

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont

# detection, angle classification and recognition

# initialize ocr engine
ocr = PaddleOCR(use_andle_cls=True, lang='en')
result = ocr.ocr(image_handwritten_path, cls=True)

# drawing result

# loading image
image = Image.open(image_handwritten_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# processing and drawing results
for res in result:
    for line in res:
        box = [tuple(point) for point in line[0]]
        # Finding the bounding box
        box = [(min(point[0] for point in box), min(point[1] for point in box)),
               (max(point[0] for point in box), max(point[1] for point in box))]
        txt = line[1][0]
        draw.rectangle(box, outline="red", width=2)  # Draw rectangle
        draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)

# saving result
image.save('result.jpg')