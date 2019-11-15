<center><font size="30"><b>Computer Vision Homework6</b></font></center>
<center><span style="font-weight:light; color:#7a7a7a; font-family:Merriweather;">by b06902034 </span><span style="font-weight:light; color:#7a7a7a; font-family:Noto Serif CJK SC;">黃柏諭</span></center>
---

### Result

<img src="/home/alec/Documents/ComputerVision/homework6/report/yokoi.png" style="zoom:67%;" />

### Code

```python
import cv2
import numpy as np

def downSample(img):
    size = int(img.shape[0] / 8)
    res = np.zeros((size, size))
    for i in range(64):
        for j in range(64):
            res[i, j] = img[i * 8, j * 8]
    return res

def getNeighborhood(img, r, c):
    res = []
    for i in range(3):
        for j in range(3):
            nr = r + i - 1
            nc = c + j - 1
            if 0 <= nr < img.shape[0] and 0 <= nc < img.shape[1]:
                res.append(img[nr][nc])
            else:
                res.append(0)
    return [res[x] for x in [4, 5, 1, 3, 7, 8, 2, 0, 6]]

def h(b, c, d, e):
    if b == c == d == e:
        return 'r'
    if b != c:
        return 's'
    return 'q'

def f(lst):
    if lst.count('r') == 4:
        return 5
    return lst.count('q')

def getYokoiMatrix(img):
    res = np.zeros(img.shape)
    for i, j in np.ndindex(img.shape):
        if img[i, j] != 0:
            n = getNeighborhood(img, i, j)
            res[i, j] = f([h(n[0], n[1], n[6], n[2]),\
                           h(n[0], n[2], n[7], n[3]),\
                           h(n[0], n[3], n[8], n[4]),\
                           h(n[0], n[4], n[5], n[1])])
    return res

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
binary = (image >= 128) * np.full(image.shape, 255)
ans = getYokoiMatrix(downSample(binary)).tolist()
with open("Output.txt", "w") as fp: 
    for i in range(len(ans)):
        for j in range(len(ans[i])):
            if ans[i][j] == 0:
                fp.write(" ")
            else:
                fp.write(str(int(ans[i][j])))
        fp.write("\n")
```

