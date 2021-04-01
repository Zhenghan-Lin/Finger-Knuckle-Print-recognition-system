import cv2 as cv
import numpy as np
import math

img = cv.imread(r"../img/sample.jpg", cv.IMREAD_GRAYSCALE)
print(img.shape[0])