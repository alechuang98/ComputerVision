import cv2
import numpy as np

image = cv2.imread("lena.bmp")
upsideDown = np.zeros(image.shape)
rightsideLeft = np.zeros(image.shape)
diagonallyMirrored = np.zeros(image.shape)

lenR = image.shape[0]
lenC = image.shape[1]
for r in range(lenR):
    for c in range(lenC):
        upsideDown[lenR - r - 1, c, :] = image[r, c, :]
        rightsideLeft[r, lenC - c - 1, :] = image[r, c, :]
        diagonallyMirrored[c, r, :] = image[r, c, :]

cv2.imwrite("1_upside-down lena.bmp", upsideDown)
cv2.imwrite("2_right-side-left lena.bmp", rightsideLeft)
cv2.imwrite("3_diagonally mirrored lena.bmp", diagonallyMirrored)
