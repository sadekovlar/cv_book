import cv2
import numpy as np

#Canny Edge Detector
#This detector uses gradients for finding edges
#First of all, Canny compute gradients for x and for y by 3x3 kernels with Sobel filter
#And after compute sum of absolute value of two gradients 

def canny_edge_detector(image, smallest_value, highest_value):
    '''
    Get image and two thesholds, smallest and highest value
    Between that two values edges are computed
    '''
    edges = cv2.Canny(image, smallest_value, highest_value)

    return edges

#Tenengead edge detector is almost the same with Canny
#But this algorithm uses square root of sum of square gradients instead sum of absolute values
#It is less accurate than Canny, but also very useful

def tenengrad_edge_detector(image):
    #dx and dy are used for showing where Sobel operator works
    #ksize is kernel size
    dx = cv2.Sobel(image, cv2.CV_32F, dx=1, dy=0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, dx=0, dy=1, ksize=3)

    edges = cv2.magnitude(dx, dy) #comdeute magnitue
    edges = cv2.convertScaleAbs(edges) #convert it to uint8 for visualization
    
    return edges


image = cv2.imread('data/road.png', 0)

edges_canny = canny_edge_detector(image, smallest_value=100, highest_value=200)
edges_tenengrad = tenengrad_edge_detector(image)

cv2.imshow('grayscale', image)
cv2.waitKey()

cv2.imshow('canny edges', edges_canny)
cv2.waitKey()

cv2.imshow('tenengrad edges', edges_tenengrad)
cv2.waitKey()
cv2.destroyAllWindows()