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
    while z.real*z.real + z.real*z.imag*z.imag <= 4 and m < max_iter:
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

def step_arr(start, stop, t_frames):
    mag_factor = 0.8
    arr = np.zeros(t_frames)
    prop = (stop - start)/(math.log2(t_frames)**mag_factor)
    for i in range(t_frames):
        arr[i] = start + (math.log2(1+i)**mag_factor)*prop
    return arr

## Parameters ##
# set #
minx_start, minx_end = -2, -1.4840
maxx_start, maxx_end = 1, -1.4841
miny_start, miny_end = -0.75, -0.000075
####
aspect_ratio = 1.0/2.0
res = 1000
n_frames = 240
fps = 24
max_iter = 1000
####

maxy_start, maxy_end = miny_start +  aspect_ratio*(maxx_start - minx_start) , miny_end +  aspect_ratio*(maxx_end - minx_end)
WIDTH = int((maxx_start - minx_start)*res)
HEIGHT = int(aspect_ratio*WIDTH)

nthreads = 32
nblocksx = WIDTH//nthreads + 1
nblocksy = HEIGHT//nthreads + 1

out = cv2.VideoWriter('output_mandle.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (WIDTH, HEIGHT))
img = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)
min_x = step_arr(minx_start, minx_end, n_frames)
max_x = step_arr(maxx_start, maxx_end, n_frames)
min_y = step_arr(miny_start, miny_end, n_frames)
max_y = step_arr(maxy_start, maxy_end, n_frames)

tic = tm.perf_counter()
for i in range(n_frames):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)
    create_fractal[(nblocksx, nblocksy), (nthreads, nthreads)](min_x[i], min_y[i], max_x[i], max_y[i], img, max_iter)
    bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    out.write(bgr.copy())
    if ( (i*100/n_frames) % 10 == 0):
        print(i*100/n_frames, "%", "done")
out.release()

time = tm.perf_counter() - tic
print(time)