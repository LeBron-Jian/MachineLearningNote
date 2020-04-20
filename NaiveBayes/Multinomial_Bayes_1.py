#_*_coding:utf-8_*_
 
import pandas as pd
import numpy as np
import cv2
import time
 
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score
 
# 获取数据
def load_data():
    # 读取csv数据
    raw_data = pd.read_csv('train.csv', header=0)
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]
    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.33, random_state=0)
    return train_X, test_X, train_y, test_y
 
# 二值化处理
def binaryzation(img):
    # 类型转化成Numpy中的uint8型
    cv_img = img.astype(np.uint8)
    # 大于50的值赋值为0，不然赋值为1
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img
 
# 训练，计算出先验概率和条件概率
def Train(trainset, train_labels):
    # 先验概率
    prior_probability = np.zeros(class_num)
    # 条件概率
    conditional_probability = np.zeros((class_num, feature_len, 2))
 
    # 计算
    for i in range(len(train_labels)):
        # 图片二值化，让每一个特征都只有0， 1 两种取值
        img = binaryzation(trainset[i])
        label = train_labels[i]
 
        prior_probability[label] += 1
        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1
 
    # 将条件概率归到 [1, 10001]
    for i in range(class_num):
        for j in range(feature_len):
            # 经过二值化后图像只有0， 1 两种取值
            pix_0 = conditional_probability[i][i][0]
            pix_1 = conditional_probability[i][j][1]
            # 计算0， 1像素点对应的条件概率
            probability_0 = (float(pix_0)/float(pix_0 + pix_1))*10000 + 1
            probability_1 = (float(pix_1)/float(pix_0 + pix_1))*10000 + 1
 
            conditional_probability[i][j][0] = probability_0
            conditional_probability[i][j][1] = probability_1
 
    return prior_probability, conditional_probability
 
# 计算概率
def calculate_probability(img, label):
    probability = int(prior_probability[label])
    for j in range(feature_len):
        probability *= int(conditional_probability[label][j][img[j]])
    return probability
 
# 预测
def Predict(testset, prior_probability, conditional_probability):
    predict = []
    # 对于每个输入的X，将后验概率最大的类作为X的类输出
    for img in testset:
        # 图像二值化
        img = binaryzation(img)
 
        max_label = 0
        max_probability = calculate_probability(img, 0)
        for j in range(1, class_num):
            probability = calculate_probability(img, j)
            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)
    return np.array(predict)
 
 
# MNIST数据集有10种labels，分别为“0，1,2，3,4,5,6,7,8,9
class_num = 10
feature_len = 784
 
if __name__ == '__main__':
    time_1 = time.time()
    train_X, test_X, train_y, test_y = load_data()
    prior_probability, conditional_probability = Train(train_X, train_y)
    test_predict = Predict(test_X, prior_probability, conditional_probability)
    score = accuracy_score(test_y, test_predict)
    print(score)