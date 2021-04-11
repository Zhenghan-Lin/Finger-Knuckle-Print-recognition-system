import cv2 as cv
import numpy as np
import math
from scipy import io as scio
from scipy import signal
# np.set_printoptions(threshold=np.inf)


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

    @staticmethod
    def getGaborKernel():        # 读取gabor核
        """
        this list is sorted from theta_0 to theta_12.
        :return: list
        """
        file = scio.loadmat(r"../../31_LDDBP/gaborfilter.mat")
        kernels = [i for i in file["filters"]]
        return kernels

    def reverseImage(self, img):
        """
        reverse the image by subtracting the intensity from 255.
        :return:
        """
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j] = 255 - img[i, j]
        return img

    def getElementApart(self, data, num, data_type, row, col):
        """
        get element separately on the third dimension of a list.
        :return: ndarray
        """
        temp = np.zeros((num, 1), dtype=data_type)
        for i in range(num):
            temp[i] = data[i][row][col]
        return temp

    def locateDirection(self, lddbp_code, convolutional_result):
        """
        locate the dominating direction and the subordinate direction.
        sort the directions.
        :return: descending order of dominating direction array and ascending order of subordinate direction array
        """
        #   do some preparation for the isolation, generate the circle code
        code_length = len(lddbp_code)
        temp_code = np.zeros((code_length+2, 1), dtype=int)
        temp_code[1:code_length+1] = lddbp_code[:]
        temp_code[code_length+1] = lddbp_code[0]
        temp_code[0] = lddbp_code[code_length-1]

        # print(lddbp_code)
        # print(convolutional_result)

        #   isolate the dominating direction and least direction
        dominating_result, subordinate_result = [], []
        for i in range(code_length):
            if temp_code[i + 1] == 1 and temp_code[i + 2] == 0:     # locate the dominating direction
                dominating_result.append([i, convolutional_result[i]])
            if temp_code[i + 1] == 0 and temp_code[i + 2] == 1:
                subordinate_result.append([i, convolutional_result[i]])
        #   sort the lists
        if len(dominating_result) > 1:
            dominating_result.sort(key=lambda list: list[1], reverse=True)
            subordinate_result.sort(key=lambda list: list[1], reverse=False)

        return dominating_result, subordinate_result

    def encode(self, convolutional_result, code_length):
        """
        compare the convolutional results and encode the lddbp codes.
        :return: ndarray
        """
        multiple_code = np.zeros((12, self.IMG_ROW, self.IMG_COL), dtype=int)
        for i in range(code_length):
            for j in range(self.IMG_ROW):
                for z in range(self.IMG_COL):
                    multiple_code[i][j][z] = convolutional_result[i+1][j][z] > convolutional_result[i][j][z]
        return multiple_code

    def LDDPEncode(self, index, convolutional_result):
        """
        utilize the index of direction and the contiguous responses to
        calculate LDDP(local discriminate direction pattern)
        :param index: the index that indicates either the dominating direction
                      or the subordinate direction,
        :param convolutional_result: the contiguous responses to the index
        :return: int ranging from 0 to 264
        """
        i = index + 1
        if convolutional_result[i+1] > convolutional_result[i-1]:
            lddp = index*2
        else:
            lddp = index*2-1

        return lddp

    def LDDBPCoding(self):
        """
        conduct the coding process of the image.
        the sequence of code is reversed, which means code[0] equals the result convoluted by theta_0 Gabor kernel.
        it can therefore be argued that the dominating direction is depends on the sequence of '10'.
        in other words, the subordinate direction relies on the sequence of '01'.
        :return: ndarray
        """
        img = cv.imread(self.IMG_NAME, cv.IMREAD_GRAYSCALE)
        gabor_kernels = self.getGaborKernel()

        # reverse the image
        img = self.reverseImage(img)

        # the process of convolution
        convolutional_result = []
        code_length = len(gabor_kernels)
        for i in range(code_length):
            convolutional_result.append(signal.convolve2d(img, gabor_kernels[i][0], mode="same"))

        # expand the result box, the length is 14(12+2), as it is a circle code.
        convolutional_result.append(convolutional_result[0])        # result[12] equals result[1]
        convolutional_result.insert(0, convolutional_result[code_length-1])   # result[0] equals result[11]

        # the process of encode
        multiple_code = self.encode(convolutional_result, code_length)

        # generate Lm and Ls
        Lm = np.zeros((self.IMG_ROW, self.IMG_COL), dtype=int)
        Ls = np.zeros((self.IMG_ROW, self.IMG_COL), dtype=int)
        for i in range(self.IMG_ROW):
            for j in range(self.IMG_COL):

                # get codes and convolutional results at (i,j)
                temp_code = self.getElementApart(multiple_code, code_length, int, i, j)
                temp_result = self.getElementApart(convolutional_result, code_length + 2, float, i, j)

                # the process of locating direction
                dominating_result, subordinate_result = self.locateDirection(temp_code, temp_result[1:code_length+1])

                # if there is no dominating direction, then pass on.
                # else generate Lm and Ls
                direction_num = len(dominating_result)
                if direction_num == 0:
                    continue
                else:   # calculate Lm
                    dominating_index1 = dominating_result[0][0]
                    intermediate_calculation = self.LDDPEncode(dominating_index1, temp_result)
                    subordinate_index1 = subordinate_result[0][0]
                    Lm[i][j] = (
                            (intermediate_calculation - 1) * (self.THETA_NUM - 1) +
                                np.mod(self.THETA_NUM + subordinate_index1 - dominating_index1, self.THETA_NUM)
                    )
                    # calculate Ls
                    if direction_num >= 2:
                        dominating_index2 = dominating_result[1][0]
                        intermediate_calculation1 = self.LDDPEncode(dominating_index2, temp_result)
                        subordinate_index2 = subordinate_result[1][0]
                        Ls[i][j] = (
                            (intermediate_calculation1 - 1) * (self.THETA_NUM - 1) +
                                np.mod(subordinate_index2 + self.THETA_NUM - subordinate_index2, self.THETA_NUM)
                        )

        # temp_code = self.getElementApart(multiple_code, code_length, int, 66, 66)
        # temp_result = self.getElementApart(convolutional_result, code_length+2, float, 66, 66)
        # dominating_result, subordinate_result = self.locateDirection(temp_code, temp_result[1:code_length+1])
        # print(dominating_result)
        # print(dominating_result[0])
        # print(type(dominating_result[0][1]))
        return Lm, Ls

    def LDDBP(self):

if __name__ == '__main__':
    test = FeatureExtraction(r"../img/negative.jpg")
    # gabor_kernels = test.generateGaborKernel()
    # gabor_kernels = test.getGaborKernel()
    code = test.lddbpCoding()
    print(code)
