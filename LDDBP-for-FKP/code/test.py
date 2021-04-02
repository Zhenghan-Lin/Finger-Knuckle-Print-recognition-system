import cv2 as cv
import numpy as np
import math
from scipy import io as scio
np.set_printoptions(threshold=np.inf)

data = scio.loadmat(r"C:\Users\HPuser\Desktop\Finger-Knuckle-Print-recognition-system\31_LDDBP\gaborfilter.mat")
print(type(data))
print(data["filters"][0][0])
print(data["filters"][0][0].shape)
