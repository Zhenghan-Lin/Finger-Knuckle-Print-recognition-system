import cv2 as cv
import numpy as np
import math
from scipy import io as scio
from scipy import signal
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.inf)

# data = scio.loadmat(r"..\..\31_LDDBP\gaborfilter.mat")
# result = [i for i in data["filters"]]

#   卷积并显示图像
# gabor = data["filters"][6][0]
# img = cv.imread(r"../img/positive.jpg", cv.IMREAD_GRAYSCALE)
#
# result1 = signal.convolve2d(img, gabor, mode="same")
# print(result1)

# 图像取反
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img[i, j] = 255 - img[i, j]
#
# cv.imshow("a", img)
# cv.waitKey(0)

# result2 = signal.convolve2d(img, gabor, mode="same")
#
# fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))
# # fig, (ax_mag) = plt.subplots(1, 1)
# ax_orig.imshow(np.absolute(result1), cmap='gray')
# ax_orig.set_title('Original')
# ax_orig.set_axis_off()
# ax_mag.imshow(np.absolute(result2), cmap='gray')
# ax_mag.set_title('Gradient magnitude')
# ax_mag.set_axis_off()
# # plt.savefig(r"../img/convelutional_result.jpg")
# plt.show()


import pandas as pd
import numpy as np
import time
import sys
from tqdm import tqdm, trange


def test(total):
    with tqdm(total=total, leave=False, unit='img') as pbar2:
        for Q_index in range(total):
            pbar2.set_description('No.%d is processed!' % Q_index)
            time.sleep(0.1)
            # action
            pbar2.update(1)


# with tqdm(total=a-1, desc='Example', leave=False, unit='img', unit_scale=True) as pbar1:
#     for P_index in range(a-1):
#         test(a - P_index)
#         time.sleep(0.5)
#         pbar1.update(1)


a = 100
with tqdm(total=a, leave=False, unit='img', unit_scale=True) as bar:
    for i in range(a):
        bar.set_description('No.%d ' % i)
        test(a)
        time.sleep(0.1)
        bar.update(1)


