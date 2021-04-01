import cv2 as cv
import numpy as np
import math


class FeatureExtration:
    def __init__(self):
        self.LAMBDA = 10.0318       # 波长
        self.BANDWIDTH = 1          # 带宽
        self.THETA_NUM = 12         # 方向数目，即LDDBP编码位数
        self.GAMMA = 1              # 长宽比，=1时为圆形
        self.PSI = np.pi            # gabor函数图像的相位差
        self.SIGMA = ((self.LAMBDA / np.pi) * math.sqrt(math.log(2) / 2) * ((math.pow(2, self.BANDWIDTH) + 1) /
                                                                    (math.pow(2, self.BANDWIDTH) - 1))   # 高斯函数的标准差
                      )
        self.IMG_ROW = 110          # ROI图像的高，即行数
        self.IMG_COL = 220          # ROI图像的宽，即列数
        self.ROW_PLUS_COL = self.IMG_ROW * self.IMG_COL
