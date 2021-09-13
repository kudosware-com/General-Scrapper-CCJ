# import cv2
# import os,argparse
# import pytesseract
# from PIL import Image
  
# #We then Construct an Argument Parser
# ap=argparse.ArgumentParser()
# ap.add_argument("-i","--image",
#                 required=True,
#                 help="Path to the image folder")
# ap.add_argument("-p","--pre_processor",
#                 default="thresh", 
#                 help="the preprocessor usage")
# args=vars(ap.parse_args())
  
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# #We then read the image with text
# images=cv2.imread(args["image"])
  
# #convert to grayscale image
# gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
  
# #checking whether thresh or blur
# if args["pre_processor"]=="thresh":
#     cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
# if args["pre_processor"]=="blur":
#     cv2.medianBlur(gray, 3)
        
      
# #memory usage with image i.e. adding image to memory
# filename = "{}.jpg".format(os.getpid())
# cv2.imwrite(filename, gray)
# text = pytesseract.image_to_string(Image.open(filename))
# os.remove(filename)
# print(text)
  
# # show the output images
# cv2.imshow("Image Input", images)
# cv2.imshow("Output In Grayscale", gray)
# cv2.waitKey(0)


import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, grayscale, Otsu's threshold
image = cv2.imread('screen.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and remove small noise
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        cv2.drawContours(opening, [c], -1, 0, -1)

# Invert and apply slight Gaussian blur
result = 255 - opening
result = cv2.GaussianBlur(result, (3,3), 0)

# Perform OCR
data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('result', result)
cv2.waitKey() 