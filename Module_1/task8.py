import cv2

IMAGE_PATH = 'image.jpg'

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

BLUE = [0, 0, 255]

replicate = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REPLICATE) #Pixels of border copy last pixels from original image
reflect = cv2.copyMakeBorder(image, 50, 50, 50, 50,cv2.BORDER_REFLECT) #Border will be mirror reflection of the border element
reflect101 = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT_101) #The same, but the last pixels didn't repeat
wrap = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_WRAP) #Add border with pixels from opposite side 
constant= cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE) #Border consists constant 'value'

cv2.imshow(replicate)
cv2.waitKey()

cv2.imshow('reflected', reflect)
cv2.waitKey()

cv2.imshow('reflected', reflect101)
cv2.waitKey()

cv2.imshow('wrapped', wrap)
cv2.waitKey()

cv2.imshow('constant blue', constant)
cv2.waitKey()
cv2.destroyAllWindows()