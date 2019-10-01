<center><font size="30"><b>Computer Vision HW3</b></font></center>
<center><span style="font-weight:light; color:#7a7a7a; font-family:Merriweather;">by b06902034 </span><span style="font-weight:light; color:#7a7a7a; font-family:Noto Serif CJK SC;">黃柏諭</span></center>
---

### Problem 1

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("1_original_lena.bmp", image)
plt.hist(image.reshape(image.size), bins = range(256))
plt.savefig("1_original_histogram.png")
```

<img src="/home/alec/Documents/ComputerVision/homework3/code/1_original_lena.bmp" style="zoom:50%;" />

<img src="/home/alec/Documents/ComputerVision/homework3/code/1_original_histogram.png" style="zoom:50%;" />

### Problem 2

```python
devide3 = image / 3
cv2.imwrite("2_devide3_lena.bmp", devide3)
plt.clf()
plt.hist(devide3.reshape(devide3.size), bins = range(256))
plt.savefig("2_devide3_histogram.png")
```

<img src="/home/alec/Documents/ComputerVision/homework3/code/2_devide3_lena.bmp" style="zoom:50%;" />

<img src="/home/alec/Documents/ComputerVision/homework3/code/2_devide3_histogram.png" style="zoom:50%;" />

### Problem 3

```python
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
```

<img src="/home/alec/Documents/ComputerVision/homework3/code/3_histo_lena.bmp" style="zoom:50%;" />

<img src="/home/alec/Documents/ComputerVision/homework3/code/3_histo_histogram.png" style="zoom:50%;" />