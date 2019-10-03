import cv2
import numpy as np
import matplotlib.pyplot as plt
inf = 10000000

L, R, D, U, mr, mc, cnt = 0, 0, 0, 0, 0, 0, 0
def can(row, col):
    return row >= 0 and row < 512 and col >= 0 and col < 512

def dfs(row, col, vis, img):
    global L, R, D, U, cnt, mr, mc
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if can(r, c) and vis[r, c] == 0 and img[r, c] != 0:
            vis[r, c] = 1
            cnt += 1
            mr += r
            mc += c
            L = min(L, c)
            R = max(R, c)
            D = min(D, r)
            U = max(U, r)
            stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

image = cv2.imread("lena.bmp")

binary = (image >= 128) * np.full(image.shape, 255)

img2D = image[:, :, 0].copy()
plt.hist(img2D.reshape(img2D.size), bins=range(256))

vis = np.zeros(img2D.shape)
connected = binary.copy().astype(np.int32)
for r in range(vis.shape[0]):
    for c in range(vis.shape[1]):
        L, R, D, U, cnt, mr, mc = inf, -inf, inf, -inf, 0, 0, 0
        dfs(r, c, vis, binary[:, :, 0])
        if cnt >= 500:
            cv2.rectangle(connected, (L, U), (R, D), (255, 0, 0), 2)
            cv2.circle(connected, (int(mc / cnt), int(mr / cnt)), 5, (0, 0, 255), -1)

cv2.imwrite("1_binary lena.bmp", binary)
plt.savefig("2_histogram.png")
cv2.imwrite("3_connected lana.bmp", connected)
