import cv2 as cv
import numpy as np
import math
from scipy import io as scio
from scipy import signal
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# data = scio.loadmat(r"..\..\31_LDDBP\gaborfilter.mat")
# print(data["filters"].shape[0])

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
a = []
b = [1,2,3,4,5,6]
for i in range(3):
    a = a + b
print(a)
# b = []
# for i in range(50):
#     b.append(i+1)
# print(b)
# a = [0,0,1,2,3,3,5,5,6,1,23,4,1,2,5,1,1,22,5,2,35,1,23,5,50,5,6,14,5,2,36,23,6,21,6,16,16,16,17,48,47]
# print(len(a))
# hist, _ = np.histogram(a, bins=b, density=True)
# print(hist)
# print(_)
# print(hist.sum())

