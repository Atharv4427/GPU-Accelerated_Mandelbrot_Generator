import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from numba import jit, cuda, njit, f8, vectorize, uint32, uint8
import time as tm

@cuda.jit(device=True)
def is_f(cx, cy, max_iter):
    z = 0
    m = 0
    c = complex(cx, cy)
    while z.real*z.real + z.imag*z.imag <= 4 and m < max_iter:
        z = z*z + c
        m += 1
    return m

@cuda.jit
def create_fractal ( min_x, min_y, max_x, max_y, im, iters):
    ht = im.shape[0]
    wd = im.shape[1]
    px = (max_x - min_x)/ wd
    py = (max_y - min_y)/ ht

    x,y = cuda.grid(2)

    if x<wd and y<ht:
        real =  min_x + x*px
        imag = min_y + y*py
        hue = is_f(real, imag, iters)
        value = 255 if hue < iters else 0
        hue = int(255*math.log10(hue + 1)/math.log10(1 + iters))
        im[y][x] = hue, 255, value



WIDTH = int(12000)
HEIGHT = int(8000)
max_iter = 1000
min_x, min_y, max_x, max_y = -2, -1, 1, 1

tic = tm.perf_counter()
img = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)
im = cuda.to_device(img)

pixels = WIDTH*HEIGHT
nthreads = 32
nblocksx = WIDTH//nthreads + 1
nblocksy = HEIGHT//nthreads + 1

create_fractal[(nblocksx, nblocksy), (nthreads, nthreads)](min_x, min_y, max_x, max_y, im, max_iter)
img = im.copy_to_host()

bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imwrite('fancy_fractal_5.png', bgr)
time = tm.perf_counter() - tic
print(time)