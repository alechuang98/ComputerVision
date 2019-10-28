<center><font size="30"><b>Computer Vision HW2</b></font></center>
<center><span style="font-weight:light; color:#7a7a7a; font-family:Merriweather;">by b06902034 </span><span style="font-weight:light; color:#7a7a7a; font-family:Noto Serif CJK SC;">黃柏諭</span></center>
---

### Problem 1
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.bmp")
binary = (image >= 128) * np.full(image.shape, 255)
cv2.imwrite("1_binary lena.bmp", binary)
```

<img src="/home/alec/Documents/ComputerVision/homework2/code/1_binary lena.bmp" style="zoom:50%;" />

### Problem 2
```python
img2D = image[:, :, 0].copy()
plt.hist(img2D.reshape(img2D.size), bins=range(256))
plt.savefig("2_histogram.png")
```

<img src="/home/alec/Documents/ComputerVision/homework2/code/2_histogram.png" style="zoom: 67%;" />

### Problem 3

```python
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
            
vis = np.zeros(img2D.shape)
connected = binary.copy().astype(np.int32)

for r in range(vis.shape[0]):
    for c in range(vis.shape[1]):
        L, R, D, U, cnt, mr, mc = inf, -inf, inf, -inf, 0, 0, 0
        dfs(r, c, vis, binary[:, :, 0])
        if cnt >= 500:
            cv2.rectangle(connected, (L, U), (R, D), (255, 0, 0), 2)
            cv2.circle(connected, (int(mc / cnt), int(mr / cnt)), 5, (0, 0, 255), -1)
            
cv2.imwrite("3_connected lana.bmp", connected)
```

dfs找出連通塊，檢查大小是否超過500，繪製長方形並用紅色圓形標示中心點。

這裡的連通是指**四連通**。

<img src="/home/alec/Documents/ComputerVision/homework2/code/3_connected lana.bmp" style="zoom:50%;" />

