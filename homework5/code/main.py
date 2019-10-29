import cv2
import numpy as np

def inn(x, y):
    return x[0] >= 0 and x[1] >= 0 and x[0] < y.shape[0] and x[1] < y.shape[1]

def dilation(origin, kernel):
    res = np.zeros(origin.shape)
    for r, c in np.ndindex(origin.shape):
        max_val = 0
        for i, j in kernel:
            if inn((r + i, j + c), origin):
                max_val = max(max_val, origin[r + i, j + c])
        for i, j in kernel:
            if inn((r + i, j + c), res):
                res[r + i, j + c] = max_val
    return res

def erosion(origin, kernel):
    res = np.zeros(origin.shape)
    for r, c in np.ndindex(origin.shape):
        min_val = 255
        for i, j in kernel:
            if inn((r + i, j + c), origin):
                min_val = min(min_val, origin[r + i, j + c])
        for i, j in kernel:
            if inn((r + i, j + c), res):
                res[r + i, j + c] = min_val
    return res

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
_35553 =\
[[-2, -1], [-2, 0], [-2, 1],\
 [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],\
 [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],\
 [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],\
 [2, -1], [2, 0], [2, 1]]

dil = dilation(image, _35553)
ero = erosion(image, _35553)
cv2.imwrite("1_dilation.bmp", dil)
cv2.imwrite("2_erosion.bmp", ero)
cv2.imwrite("3_opening.bmp", dilation(ero, _35553))
cv2.imwrite("4_closing.bmp", erosion(dil, _35553))