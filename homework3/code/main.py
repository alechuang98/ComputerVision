import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("1_original_lena.bmp", image)
plt.hist(image.reshape(image.size), bins = range(256))
plt.savefig("1_original_histogram.png")

devide3 = image / 3
cv2.imwrite("2_devide3_lena.bmp", devide3)
plt.clf()
plt.hist(devide3.reshape(devide3.size), bins = range(256))
plt.savefig("2_devide3_histogram.png")

cnt = [0] * 255
for x in devide3.reshape(devide3.size):
    cnt[int(x)] += 1
for i in range(1, len(cnt)):
    cnt[i] += cnt[i - 1]
histoList = [255 * cnt[int(x)] / devide3.size for x in devide3.reshape(devide3.size)]
histoEq = np.array(histoList).reshape(512, 512)
cv2.imwrite("3_histo_lena.bmp", histoEq)
plt.clf()
plt.hist(histoList, bins = range(256))
plt.savefig("3_histo_histogram.png")