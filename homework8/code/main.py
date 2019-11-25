import cv2
import numpy as np
import math

def inside(r, c):
    return 0 <= r < 512 and 0 <= c < 512

def gaussion(img, amp):
    res = np.zeros(img.shape)
    for r, c in np.ndindex(img.shape):
        res[r, c] = min(img[r, c] + int(amp * np.random.normal(0, 1)), 255)
    return res

def salt_and_pepper(img, prob):
    res = np.copy(img)
    for r, c in np.ndindex(res.shape):
        rng = np.random.uniform(0, 1)
        if rng < prob:
            res[r, c] = 0
        if rng > 1 - prob:
            res[r, c] = 255
    return res

def box(img, length):
    res = np.zeros(img.shape)
    delta = int(length / 2)
    for r, c in np.ndindex(res.shape):
        cnt, total = 0, 0
        for dr in range(-delta, delta + 1):
            for dc in range(-delta, delta + 1):
                if inside(r + dr, c + dc):
                    cnt += 1
                    total += img[r + dr, c + dc]
        res[r, c] = int(total / cnt)
    return res

def median(img, length):
    res = np.zeros(img.shape)
    delta = int(length / 2)
    for r, c in np.ndindex(res.shape):
        total = []
        for dr in range(-delta, delta + 1):
            for dc in range(-delta, delta + 1):
                if inside(r + dr, c + dc):
                    total.append(img[r + dr, c + dc])
        res[r, c] = int(np.median(total))
    return res

def dilation(origin, kernel):
    res = np.zeros(origin.shape)
    for r, c in np.ndindex(origin.shape):
        max_val = 0
        for i, j in kernel:
            if inside(r + i, j + c):
                max_val = max(max_val, origin[r + i, j + c])
        res[r, c] = max_val
    return res

def erosion(origin, kernel):
    res = np.zeros(origin.shape)
    for r, c in np.ndindex(origin.shape):
        min_val = 255
        for i, j in kernel:
            if inside(r + i, j + c):
                min_val = min(min_val, origin[r + i, j + c])
        res[r, c] = min_val
    return res

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)
        
def getSNR(origin, noise):
    u_s, vs, u_n, vn = 0, 0, 0, 0
    for r, c in np.ndindex(origin.shape):
        u_s += origin[r, c]
    u_s /= float(origin.size)
    for r, c in np.ndindex(origin.shape):
        vs += (int(origin[r, c]) - u_s) ** 2
    vs /= float(origin.size)
    for r, c in np.ndindex(origin.shape):
        u_n += float(int(noise[r, c]) - int(origin[r, c]))
    u_n /= origin.size
    for r ,c in np.ndindex(origin.shape):
        vn += float(int(noise[r, c]) - int(origin[r, c]) - u_n) ** 2
    vn /= origin.size
    return 20 * math.log((vs ** .5) / (vn ** .5), 10)

_35553 =\
[[-2, -1], [-2, 0], [-2, 1],\
 [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],\
 [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],\
 [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],\
 [2, -1], [2, 0], [2, 1]]

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

# meaningless homework :(
gaussion_10 = gaussion(image, 10)
gaussion_30 = gaussion(image, 30)
salt_and_pepper_01 = salt_and_pepper(image, .1)
salt_and_pepper_005 = salt_and_pepper(image, .05)
box_3x3_gaussion_10 = box(gaussion_10, 3)
box_5x5_gaussion_10 = box(gaussion_10, 5)
box_3x3_gaussion_30 = box(gaussion_30, 3)
box_5x5_gaussion_30 = box(gaussion_30, 5)
box_3x3_salt_and_pepper_01 = box(salt_and_pepper_01, 3)
box_5x5_salt_and_pepper_01 = box(salt_and_pepper_01, 5)
box_3x3_salt_and_pepper_005 = box(salt_and_pepper_005, 3)
box_5x5_salt_and_pepper_005 = box(salt_and_pepper_005, 5)
median_3x3_gaussion_10 = median(gaussion_10, 3)
median_5x5_gaussion_10 = median(gaussion_10, 5)
median_3x3_gaussion_30 = median(gaussion_30, 3)
median_5x5_gaussion_30 = median(gaussion_30, 5)
median_3x3_salt_and_pepper_01 = median(salt_and_pepper_01, 3)
median_5x5_salt_and_pepper_01 = median(salt_and_pepper_01, 5)
median_3x3_salt_and_pepper_005 = median(salt_and_pepper_005, 3)
median_5x5_salt_and_pepper_005 = median(salt_and_pepper_005, 5)
co_gaussion_10 = closing(opening(gaussion_10, _35553), _35553)
oc_gaussion_10 = opening(closing(gaussion_10, _35553), _35553)
co_gaussion_30 = closing(opening(gaussion_30, _35553), _35553)
oc_gaussion_30 = opening(closing(gaussion_30, _35553), _35553)
co_salt_and_pepper_01 = closing(opening(salt_and_pepper_01, _35553), _35553)
oc_salt_and_pepper_01 = opening(closing(salt_and_pepper_01, _35553), _35553)
co_salt_and_pepper_005 = closing(opening(salt_and_pepper_005, _35553), _35553)
oc_salt_and_pepper_005 = opening(closing(salt_and_pepper_005, _35553), _35553)

cv2.imwrite("gaussion_10.bmp", gaussion_10)
cv2.imwrite("gaussion_30.bmp", gaussion_30)
cv2.imwrite("salt_and_pepper_01.bmp", salt_and_pepper_01)
cv2.imwrite("salt_and_pepper_005.bmp", salt_and_pepper_005)
cv2.imwrite("box_3x3_gaussion_10.bmp", box_3x3_gaussion_10)
cv2.imwrite("box_5x5_gaussion_10.bmp", box_5x5_gaussion_10)
cv2.imwrite("box_3x3_gaussion_30.bmp", box_3x3_gaussion_30)
cv2.imwrite("box_5x5_gaussion_30.bmp", box_5x5_gaussion_30)
cv2.imwrite("box_3x3_salt_and_pepper_01.bmp", box_3x3_salt_and_pepper_01)
cv2.imwrite("box_5x5_salt_and_pepper_01.bmp", box_5x5_salt_and_pepper_01)
cv2.imwrite("box_3x3_salt_and_pepper_005.bmp", box_3x3_salt_and_pepper_005)
cv2.imwrite("box_5x5_salt_and_pepper_005.bmp", box_5x5_salt_and_pepper_005)
cv2.imwrite("median_3x3_gaussion_10.bmp", median_3x3_gaussion_10)
cv2.imwrite("median_5x5_gaussion_10.bmp", median_5x5_gaussion_10)
cv2.imwrite("median_3x3_gaussion_30.bmp", median_3x3_gaussion_30)
cv2.imwrite("median_5x5_gaussion_30.bmp", median_5x5_gaussion_30)
cv2.imwrite("median_3x3_salt_and_pepper_01.bmp", median_3x3_salt_and_pepper_01)
cv2.imwrite("median_5x5_salt_and_pepper_01.bmp", median_5x5_salt_and_pepper_01)
cv2.imwrite("median_3x3_salt_and_pepper_005.bmp", median_3x3_salt_and_pepper_005)
cv2.imwrite("median_5x5_salt_and_pepper_005.bmp", median_5x5_salt_and_pepper_005)
cv2.imwrite("co_gaussion_10.bmp", co_gaussion_10)
cv2.imwrite("oc_gaussion_10.bmp", oc_gaussion_10)
cv2.imwrite("co_gaussion_30.bmp", co_gaussion_30)
cv2.imwrite("oc_gaussion_30.bmp", oc_gaussion_30)
cv2.imwrite("co_salt_and_pepper_01.bmp", co_salt_and_pepper_01)
cv2.imwrite("oc_salt_and_pepper_01.bmp", oc_salt_and_pepper_01)
cv2.imwrite("co_salt_and_pepper_005.bmp", co_salt_and_pepper_005)
cv2.imwrite("oc_salt_and_pepper_005.bmp", oc_salt_and_pepper_005)

print(getSNR(image, gaussion_10))
print(getSNR(image, gaussion_30))
print(getSNR(image, salt_and_pepper_01))
print(getSNR(image, salt_and_pepper_005))
print(getSNR(image, box_3x3_gaussion_10))
print(getSNR(image, box_5x5_gaussion_10))
print(getSNR(image, box_3x3_gaussion_30))
print(getSNR(image, box_5x5_gaussion_30))
print(getSNR(image, box_3x3_salt_and_pepper_01))
print(getSNR(image, box_5x5_salt_and_pepper_01))
print(getSNR(image, box_3x3_salt_and_pepper_005))
print(getSNR(image, box_5x5_salt_and_pepper_005))
print(getSNR(image, median_3x3_gaussion_10))
print(getSNR(image, median_5x5_gaussion_10))
print(getSNR(image, median_3x3_gaussion_30))
print(getSNR(image, median_5x5_gaussion_30))
print(getSNR(image, median_3x3_salt_and_pepper_01))
print(getSNR(image, median_5x5_salt_and_pepper_01))
print(getSNR(image, median_3x3_salt_and_pepper_005))
print(getSNR(image, median_5x5_salt_and_pepper_005))
print(getSNR(image, co_gaussion_10))
print(getSNR(image, oc_gaussion_10))
print(getSNR(image, co_gaussion_30))
print(getSNR(image, oc_gaussion_30))
print(getSNR(image, co_salt_and_pepper_01))
print(getSNR(image, oc_salt_and_pepper_01))
print(getSNR(image, co_salt_and_pepper_005))
print(getSNR(image, oc_salt_and_pepper_005))