import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=np.inf)


class Evaluation:
    def __init__(self, filepath, threshold):
        """
        Initialize essential parameters.
        :param filepath: path of descriptor list file
        :param threshold: the threshold of classification
        """
        self.IMG_ROW = 110              # ROI图像的高，即行数
        self.IMG_COL = 220              # ROI图像的宽，即列数
        self.ROW_Multiply_COL = self.IMG_ROW * self.IMG_COL     # 行数*列数
        self.BLOCK_SIZE = 16            # 分块大小
        self.BLOCK_NUM = math.floor(self.IMG_ROW / self.BLOCK_SIZE) * \
                         math.floor(self.IMG_COL / self.BLOCK_SIZE)         # 分块总数
        self.FLAG_FOR_MINIBATCH = True      # True, use the mini batch, vice versa.
        self.THRESHOLD = threshold
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.G_TOTAL, self.I_TOTAL = 0, 0
        print("Start Loading descriptor list!")
        self.DESCRIPTOR_LIST = np.load(filepath)
        print("loading process done!")

    def match(self, sample_P, sample_Q):
        """
        the match method based on Chi-square distance.
        :return:
        """
        # calculate the square difference between img P and Q.
        difference = np.subtract(sample_P, sample_Q)
        square_difference = np.square(difference)

        # calculate the sum of P and Q
        sum = np.add(sample_P, sample_Q)

        # calculate the quotient. significantly, the denominator cannot be 0.
        index_of_nonzero = np.nonzero(sum)
        quotient = 0
        for i in range(len(index_of_nonzero[0])):
            index = index_of_nonzero[0][i]
            quotient += np.divide(square_difference[index], sum[index])

        score = quotient / self.BLOCK_NUM
        return score

    def evaluate(self):
        """
        Evaluation procedure.
        :return:
        """
        if self.FLAG_FOR_MINIBATCH:     # use mini batch for evaluation.
            picture_number = 48
        else:                           # use universal set for evaluation.
            picture_number = len(self.DESCRIPTOR_LIST)

        # evaluation start!
        for P in range(picture_number):
            for Q in range(P+1, picture_number):
                # calculate the group of P, within which is the positive sample.
                range_start = P // 12 * 12
                range_end = range_start + 11
                # calculate the score
                score = self.match(P, Q)
                # Q is one of the group P and score is less than threshold, then it is TP
                if score <= self.THRESHOLD and range_start <= Q <= range_end:




if __name__ == '__main__':
    tester = Evaluation(r'./descriptor_list/minibatch_descriptor_.npy', 1.400)
    tester.evaluate()
