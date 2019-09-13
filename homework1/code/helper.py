import cv2
import numpy as np

image = cv2.imread("lena.bmp")
h, w = image.shape[:2]
center = (h / 2, w / 2)
Mat = cv2.getRotationMatrix2D(center, 45, 1)
cv2.imwrite("4_rotate45.bmp", cv2.warpAffine(image, Mat, (h, w)))
cv2.imwrite("5_shrink.bmp", cv2.resize(image, (h >> 1, w >> 1))) 
cv2.imwrite("6_binary.bmp", cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1])