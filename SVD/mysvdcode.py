#_*_ coding:utf-8_*_
import numpy as np
import cv2
 
img = cv2.imread('harden.jpg')
print('origin image shape is ', img.shape)
# 表示 RGB 中各有一个矩阵，都为300*532
#  origin image shape is  (300, 532, 3)
 
 
def svd_compression(img, k):
    res_image = np.zeros_like(img)
    for i in range(img.shape[2]):
        # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
        U, Sigma, VT = np.linalg.svd(img[:,:,i])
        res_image[:, :, i] = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
 
    return res_image
 
 
# 保留前 k 个奇异值
res1 = svd_compression(img, k=300)
res2 = svd_compression(img, k=200)
res3 = svd_compression(img, k=100)
res4 = svd_compression(img, k=50)
 
row11 = np.hstack((res1, res2))
row22 = np.hstack((res3, res4))
res = np.vstack((row11, row22))
 
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
