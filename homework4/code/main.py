import cv2
import numpy as np
import matplotlib.pyplot as plt

def add(x, y):
    res = set()
    for v1 in x:
        for v2 in y:
            res.add(v1 + v2)
    return list(res)

def inn(x, y):
    return x[0] >= 0 and x[1] >= 0 and x[0] < y.shape[0] and x[1] < y.shape[1]

def dilation(origin, kernel):
    lst = []
    for i, j in np.ndindex(origin.shape):
        if origin[i, j] == 255:
            lst.append(np.array([i, j]))
    res = np.zeros(origin.shape)
    for i, j in add(lst, kernel):
        if i < res.shape[0] and j < res.shape[1]:
            res[i, j] = 255
    return res

def erosion(origin, kernel):
    res = np.zeros(origin.shape)
    for i, j in np.ndindex(origin.shape):
        res = add([np.array([i, j])], kernel)
        origin[i, j] = 255
        for i, j in res:
            res[i, j] = 255
            for v in add([np.array([i, j])], kernel):
                if not inn(v, res) or origin[v[0], v[1]] == 0:
                    res[i, j] = 0
        

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
test = np.array([1, 2])
print(test[0])
binary = (image >= 128) * np.full(image.shape, 255)

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