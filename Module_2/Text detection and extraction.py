#Task: detecting and extracting text from a color image

#pip install pytesseract
#download tesseract.exe from https://github.com/UB-Mannheim/tesseract/wiki

#importing libraries
import cv2
from pytesseract import pytesseract

#path to the tesseract binary file
path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.tesseract_cmd = path_to_tesseract

#reading an image
img = cv2.imread('1.png')
#conversion to black and white
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#convert a halftone image to binary
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow('threshold_image',thresh1)
cv2.waitKey()
cv2.destroyAllWindows()

#the method of structural elements with a kernel size depending on the area of the text
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#binary image expansion method (to obtain text borders)
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 6)
cv2.imshow('dilation_image.jpg',dilation)
cv2.waitKey()
cv2.destroyAllWindows()

#using the findContours method (to get the area of white pixels)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

im2 = img.copy()

#get the coordinates of the area of white pixels and draw a bounding box around it
#also save the text with the image to a text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    #draw a bounding box on the text area
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #crop the area of the bounding box
    cropped = im2[y:y + h, x:x + w]

    #opening a text file
    file = open("Output.txt", "a")

    #using tesseract on a cropped image area to get text
    text = pytesseract.image_to_string(cropped)

    #adding text to the file
    file.write(text)
    file.write("\n")

    #closing the file
    file.close

cv2.imshow('rectanglebox.jpg',rect)
cv2.waitKey()
cv2.destroyAllWindows()

