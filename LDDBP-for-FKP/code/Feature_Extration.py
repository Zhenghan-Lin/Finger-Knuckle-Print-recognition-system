import cv2 as cv
import numpy as np
import math
from scipy import io as scio
from scipy import signal
np.set_printoptions(threshold=np.inf)


class FeatureExtraction:
    def __init__(self, name=""):
        self.LAMBDA = 9.9932       # 波长
        # self.LAMBDA = 10.0318       # 波长

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
        self.IMG_NAME = name
        # print(self.SIGMA)

    def generateGaborKernel(self):      # 计算gabor核
        gabor_kernels = []
        for i in range(self.THETA_NUM):
            theta = i * np.pi / self.THETA_NUM
            gabor_kernels.append(cv.getGaborKernel((35, 35), self.SIGMA, theta, self.LAMBDA, self.GAMMA,
                                                   self.PSI, cv.CV_32F))
        return gabor_kernels

    def getGaborKernel(self):        # 读取gabor核
        file = scio.loadmat(r"../../31_LDDBP/gaborfilter.mat")
        kernels = [i for i in file["filters"]]
        return kernels

    def lddbp_coding(self):         # 对图片进行LDDBP特征编码
        """
        conduct the coding process of the image.
        :return:
        """
        img = cv.imread(self.IMG_NAME, cv.IMREAD_GRAYSCALE)
        gabor_kernels = self.getGaborKernel()
        # the process of convolution
        convolutional_result = []
        code_length = len(gabor_kernels)
        for i in range(code_length):     # 对图像进行卷积操作
            convolutional_result.append(signal.convolve2d(img, gabor_kernels[i][0], mode="same"))
        # expand the result box, the length is 14(12+2), as it is a circle code.
        convolutional_result.append(convolutional_result[0])        # result[12] equals result[1]
        convolutional_result.insert(0, convolutional_result[code_length])   # result[0] equals result[11]


if __name__ == '__main__':
    test = FeatureExtraction()
    # gabor_kernels = test.generateGaborKernel()
    gabor_kernels = test.getGaborKernel()
    print(gabor_kernels)
    print(type(gabor_kernels))
    print(type(gabor_kernels[0]))
    print(gabor_kernels[0])
