import cv2
import numpy as np
from scipy import signal
import math

def crossCorrelation(img, msk):
    pad_width = int(msk.shape[0] / 2)
    return signal.fftconvolve(np.pad(img, (pad_width, pad_width), "edge"), np.flip(msk), mode = "valid").round(0).astype(int)
    
def zeroCross(img, msk, threshold):
    mat = crossCorrelation(img, msk)
    res = np.full(img.shape, 255)
    for r, c in np.ndindex(img.shape):
        if mat[r, c] < threshold:
            continue
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if 0 <= r + dr < img.shape[0] and 0 <= c + dc < img.shape[1]\
                and mat[r + dr, c + dc] <= -threshold:
                    res[r, c] = 0
    return res


laplacian1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) / 3
min_laplacian = np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]) / 3
log = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],\
                [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
                [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
                [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
                [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
                [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],\
                [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],\
                [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],\
                [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],\
                [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],\
                [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
dog = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],\
                [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
                [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
                [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
                [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
                [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],\
                [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],\
                [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],\
                [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],\
                [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],\
                [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("1_laplacian.bmp", zeroCross(image, laplacian1, 15))
cv2.imwrite("2_laplacian.bmp", zeroCross(image, laplacian2, 15))
cv2.imwrite("3_min_laplacian.bmp", zeroCross(image, min_laplacian, 20))
cv2.imwrite("4_log.bmp", zeroCross(image, log, 3000))
cv2.imwrite("5_dog.bmp", zeroCross(image, dog, 1))

