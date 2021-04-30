import cv2 as cv
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=np.inf)


class Evaluation:
    def __init__(self, filepath, threshold: float, gabor, blocksize, reversal, flag_for_minibatch: bool):
        """
        Initialize essential parameters.
        :param filepath: path of descriptor list file
        :param threshold: the threshold of classification
        :param gabor: the version of Gabor kernels
        :param blocksize: the size of block
        :param reversal: flag of reversal
        :param flag_for_minibatch: flag of the use of minibatch
        """
        print("Start Loading descriptor list!")
        self.DESCRIPTOR_LIST = np.load(filepath)
        print("loading process done!")
        self.THRESHOLD = threshold          # threshold for classification
        self.GABOR_VERSION = gabor          # version of gabor kernels using for file title.
        self.REVERSAL = reversal            # flag of reversal, using for file title.
        self.FLAG_FOR_MINIBATCH = flag_for_minibatch      # True, use the mini batch, vice versa.
        self.BLOCK_SIZE = blocksize            # 分块大小

        self.IMG_ROW = 110              # ROI图像的高，即行数
        self.IMG_COL = 220              # ROI图像的宽，即列数
        self.ROW_Multiply_COL = self.IMG_ROW * self.IMG_COL     # 行数*列数
        self.BLOCK_NUM = math.floor(self.IMG_ROW / self.BLOCK_SIZE) * math.floor(self.IMG_COL / self.BLOCK_SIZE)
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0     # sum of TP/TN/FP/FN
        self.FP_LIST, self.FN_LIST = [], []                 # storage of mis-classification
        self.GENUINE, self.IMPOSTER = [], []                # storage of each category
        # todo 初始化统计nan的工具
        self.NAN_LIST = []      # collect abnormal information of nan
        self.NAN_COUNT = 0

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
                    # todo 统计nan信息
                    if np.isnan(score):
                        self.NAN_LIST.append((P_index, Q_index, score))
                        self.NAN_COUNT += 1

                    # whether Q belongs to group P
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

        # calculate some evaluating indicators
        Genuine = self.TP+self.FN
        Imposter = self.TN+self.FP
        Total = self.TP+self.TN+self.FP+self.FN
        Precision = self.TP / (self.TP + self.FP)
        Recall = self.TP / (self.TP + self.FN)
        FPR = self.FP / (self.FP + self.TN)
        TPR = self.TP / (self.TP + self.FN)
        F1_measure = 2 * Precision * Recall / (Precision + Recall)

        # save data
        suffix = str(self.GABOR_VERSION)+'_block'+str(self.BLOCK_SIZE)+'_'+str(self.REVERSAL)
        np.savetxt(r'./Genuine/Genuine_'+suffix+'.txt', self.GENUINE, delimiter='\t', fmt='%f')
        np.savetxt(r'./Imposter/Imposter_'+suffix+'.txt', self.IMPOSTER, delimiter='\t', fmt='%f')
        np.savetxt(r'./FP_list/FP_list_'+suffix+'.txt', self.FP_LIST, delimiter='\t', fmt='%f')
        np.savetxt(r'./FN_list/FN_list_'+suffix+'.txt', self.FN_LIST, delimiter='\t', fmt='%f')
        df = pd.DataFrame(
            {'Threshold': [self.THRESHOLD], 'Precision': [Precision],
             'Recall': [Recall], 'TPR': [TPR], 'FPR': [FPR], 'F1-measure': [F1_measure]},
            columns=['Threshold', 'Precision', 'Recall', 'TPR', 'FPR', 'F1-measure']
        )
        df.to_csv(r'./result/result_'+suffix+'.csv', mode='a+', index=False, sep='\t', float_format='%.6f')
        # todo 将nan信息写入磁盘
        np.savetxt(r'./Nan/Nan_'+suffix+'.txt', self.NAN_LIST, delimiter='\t', fmt='%f')

        # print results
        print('saving process finished!')
        print('Genuine: {:d} \t Imposter: {:d} \t Total: {:d}'.format(Genuine, Imposter, Total))
        print('TP: {:d} \t FN: {:d} \t TN: {:d} \t FP: {:d}'.format(self.TP, self.FN, self.TN, self.FP))
        print('Precision: {:%}'.format(Precision))
        print('Recall: {:%}'.format(Recall))
        print('TPR: {:%} '.format(TPR))
        print('FPR: {:%} '.format(FPR))
        print('F1-measure: {:f}'.format(F1_measure))
        # todo 打印提示信息
        if self.NAN_COUNT > 0:
            print('\033[31mWarning: \033[0m')
            print('\033[31mWarning: Nan has occurred {:d} times, '
                  'please check files for more details. \033[0m'.format(self.NAN_COUNT))


if __name__ == '__main__':
    # todo 测试最佳分类阈值
    # todo 测试时的得分，和评价时的得分差别很大，查查看
    descriptor_path = r'descriptor_list/Minibatch_descriptor_1_block8_True.npy'
    threshold = 0.3295
    Gabor = 1
    Block_size = 8
    Reversal = True
    flag_for_minibatch = True
    tester = Evaluation(descriptor_path, threshold, Gabor, Block_size, Reversal, flag_for_minibatch)
    tester.evaluate()


