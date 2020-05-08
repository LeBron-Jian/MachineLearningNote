from  numpy  import *
# li = [[1,1],[1,3],[2,3],[4,4],[2,4]]
li = [[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]]
matrix = mat(li)
# 求均值
mean_matrix = mean(matrix,axis=0)
# print(mean_matrix)    #[[1.81 1.91]]
# 减去平均值
Dataadjust = matrix - mean_matrix
# print(Dataadjust)
'''
[[ 0.69  0.49]
 [-1.31 -1.21]
 [ 0.39  0.99]
 [ 0.09  0.29]
 [ 1.29  1.09]
 [ 0.49  0.79]
 [ 0.19 -0.31]
 [-0.81 -0.81]
 [-0.31 -0.31]
 [-0.71 -1.01]]
'''
#计算特征值和特征向量
covMatrix = cov(Dataadjust,rowvar=0)
# print(covMatrix)
'''
[[0.61655556 0.61544444]
 [0.61544444 0.71655556]]
'''
eigValues , eigVectors = linalg.eig(covMatrix)
# print(eigValues)
# print(eigVectors)
'''
[0.0490834  1.28402771]

[[-0.73517866 -0.6778734 ]
 [ 0.6778734  -0.73517866]]'''
# 对特征值进行排序
eigValuesIndex = argsort(eigValues)
# print(eigValuesIndex)
# 保留前K个最大的特征值
eigValuesIndex = eigValuesIndex[:-(1000000):-1]
# print(eigValuesIndex)
# 计算出对应的特征向量
trueEigVectors = eigVectors[:,eigValuesIndex]
print(trueEigVectors)
'''
[[-0.6778734  -0.73517866]
 [-0.73517866  0.6778734 ]]
 '''
# # 选择较大特征值对应的特征向量
maxvector_eigval = trueEigVectors[:,0]
print(maxvector_eigval)
'''
[-0.6778734  -0.73517866]
'''
# # 执行PCA变换：Y=PX 得到的Y就是PCA降维后的值 数据集矩阵
pca_result = maxvector_eigval * Dataadjust.T
print(pca_result)
'''
[[-0.82797019  1.77758033 -0.99219749 -0.27421042 -1.67580142 -0.9129491
   0.09910944  1.14457216  0.43804614  1.22382056]]
'''