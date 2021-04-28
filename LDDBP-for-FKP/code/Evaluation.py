import cv2 as cv
import numpy as np
import math
from tqdm import tqdm
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
        self.THRESHOLD = threshold          # threshold for classification
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0     # sum of TP/TN/FP/FN
        self.FP_LIST, self.FN_LIST = [], []                 # storage of mis-classification
        self.GENUINE, self.IMPOSTER = [], []                # storage of each category
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
        :return: none
        """
        if self.FLAG_FOR_MINIBATCH:     # use mini batch for evaluation.
            picture_number = 48
        else:                           # use universal set for evaluation.
            picture_number = len(self.DESCRIPTOR_LIST)

        # evaluation start!
        with tqdm(total=picture_number-1, unit='img', unit_scale=True) as pbar:
            for P_index in range(picture_number-1):
                pbar.set_description('No. %d' % P_index)
                for Q_index in range(P_index+1, picture_number):
                    # calculate the group of P, within which is the positive sample.
                    range_start = P_index // 12 * 12
                    range_end = range_start + 11

                    # calculate the score
                    score = self.match(self.DESCRIPTOR_LIST[P_index], self.DESCRIPTOR_LIST[Q_index])

                    # whether Q belongs to group P
                    # TODO 统计错误数据，以此确定相似度阈值
                    # TODO 统计Genuine/Imposter各个分数的数量
                    if range_start <= Q_index <= range_end:
                        # Genuine
                        self.GENUINE.append((P_index, Q_index, score, self.THRESHOLD))
                        if score <= self.THRESHOLD:     # classified as positive
                            self.TP += 1
                        else:                           # classified as negative
                            self.FN += 1
                            self.FN_LIST.append((P_index, Q_index, score, self.THRESHOLD))
                    else:
                        # Imposter
                        self.IMPOSTER.append((P_index, Q_index, score, self.THRESHOLD))
                        if score <= self.THRESHOLD:     # classified as positive
                            self.FP += 1
                            self.FP_LIST.append((P_index, Q_index, score, self.THRESHOLD))
                        else:                           # classified as negative
                            self.TN += 1
                pbar.update(1)
        # save data
        """original"""
        # np.savetxt(r'./Genuine/Genuine_.txt', self.GENUINE, delimiter='\t', fmt='%f')
        # np.savetxt(r'./Imposter/Imposter_.txt', self.IMPOSTER, delimiter='\t', fmt='%f')
        # np.savetxt(r'./FP_list/FP_list_.txt', self.FP_LIST, delimiter='\t', fmt='%f')
        # np.savetxt(r'./FN_list/FN_list_.txt', self.FN_LIST, delimiter='\t', fmt='%f')
        """9.9932"""
        np.savetxt(r'./Genuine/Genuine_9.txt', self.GENUINE, delimiter='\t', fmt='%f')
        np.savetxt(r'./Imposter/Imposter_9.txt', self.IMPOSTER, delimiter='\t', fmt='%f')
        np.savetxt(r'./FP_list/FP_list_9.txt', self.FP_LIST, delimiter='\t', fmt='%f')
        np.savetxt(r'./FN_list/FN_list_9.txt', self.FN_LIST, delimiter='\t', fmt='%f')
        """10.0318"""
        # np.savetxt(r'./Genuine/Genuine_10.txt', self.GENUINE, delimiter='\t', fmt='%f')
        # np.savetxt(r'./Imposter/Imposter_10.txt', self.IMPOSTER, delimiter='\t', fmt='%f')
        # np.savetxt(r'./FP_list/FP_list_10.txt', self.FP_LIST, delimiter='\t', fmt='%f')
        # np.savetxt(r'./FN_list/FN_list_10.txt', self.FN_LIST, delimiter='\t', fmt='%f')
        print('saving process finished!')
        print('TP: {:d} \t FN: {:d} \t TN: {:d} \t FP: {:d}'.format(self.TP, self.FN, self.TN, self.FP))
        print('Precision: {:%}'.format(self.TP / (self.TP + self.FP)))
        print('Recall: {:%}'.format(self.TP / (self.TP + self.FN)))
        print('FPR: {:%} '.format(self.FP / (self.FP + self.TN)))
        print('TPR: {:%} '.format(self.TP / (self.TP + self.FN)))


if __name__ == '__main__':
    # todo 测试最佳分类阈值
    # tester = Evaluation(r'./descriptor_list/minibatch_descriptor_.npy', 1.400)
    tester = Evaluation(r'./descriptor_list/minibatch_descriptor_9_9932.npy', 1.380)
    # tester = Evaluation(r'./descriptor_list/minibatch_descriptor_10_0318.npy', 1.400)
    tester.evaluate()
