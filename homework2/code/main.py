import cv2
import numpy as np
import matplotlib.pyplot as plt
inf = 10000000

l, r, d, u
def dfs(row, col, vis, img):
    global l, r, d, u
    if row < 0 or row > 511 or col < 0 or col > 511 or vis[row, col] == 1 or img[row, col] != 0:
        return
    l = min(l, col)
    r = max(r, col)
    d = min(d, row)
    u = max(u, row)
    dfs(row + 1, col, vis, img)
    dfs(row - 1, col, vis, img)
    dfs(row, col + 1, vis, img)
    dfs(row, col - 1, vis, img)

image = cv2.imread("lena.bmp")
binary = (image >= 128) * np.full(image.shape, 255)
img2D = image[:, :, 0].copy()
plt.hist(img2D.reshape(img2D.size), bins=range(256))

vis = np.zeros(img2D.shape)

for r in range(vis.shape[0]):
    for c in range(vis.shape[1]):
        if vis[r, c] == 0:
            l, r, d, u = inf, -inf, inf, -inf
            dfs(r, c, vis, binary[:, :, 0])
            

cv2.imwrite("1_binary lena.bmp", binary)
plt.savefig("2_histogram.png")
