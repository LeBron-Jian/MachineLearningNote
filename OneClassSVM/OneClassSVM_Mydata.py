import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
 
A = []
AA = []
 
def get_files(file_dir):
    for file in os.listdir(file_dir + '/AA475_B25'):
       A.append(file_dir + '/AA475_B25/' + file)
    length_A = len(os.listdir(file_dir + '/AA475_B25'))
    for file in range(length_A):
        img = Image.open(A[file])
        new_img = img.resize((128, 128))
        new_img = new_img.convert("L")
        matrix_img = np.asarray(new_img)
        AA.append(matrix_img.flatten())
    images1 = np.matrix(AA)
    return images1
 
def OneClassSVM_train(data1):
    from sklearn import svm
 
    trainSet = data1
    # nu 是异常点比例，默认是0.5
    clf = svm.OneClassSVM(nu=0.05, kernel='linear', gamma=0.1)
    clf.fit(trainSet)
    y_pred_train = clf.predict(trainSet)
    normal = trainSet[y_pred_train == 1]
    abnormal = trainSet[y_pred_train == -1]
    print(normal)
    print(abnormal)
    print(normal.shape)
    print(abnormal.shape)
    plt.plot(normal[:, 0], normal[:, 1], 'bx')
    plt.plot(abnormal[:, 0], abnormal[:, 1], 'ro')
    plt.show()
 
 
if __name__ == '__main__':
    train_dir = '../one_variable/train'
    data = get_files(train_dir)
    OneClassSVM_train(data)