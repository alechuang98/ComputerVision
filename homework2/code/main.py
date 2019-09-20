import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.bmp")
binary = (image >= 128) * np.full(image.shape, 255)
histo = image[:, :, 0].copy()
plt.hist(histo.reshape(histo.size), bins=range(256))

lenR = image.shape[0]
lenC = image.shape[1]

cv2.imwrite("1_binary lena.bmp", binary)
