import numpy as np

'''
mat() 函数与array()函数生成矩阵所需的数据格式有区别
mat()函数中数据可以为字符串以分号（;）分割
或者为列表形式以逗号（，）分割

array()函数中数据只能为后者形式
'''
# import numpy as np
#
# a = np.mat('1 3;5 7')
# b = np.mat([[1,2],[3,4]])
# print(a)
# print(b)
# print(type(a))
# print(type(b))
# c = np.array([[1,3],[4,5]])
# print(c)
# print(type(c))


# import numpy as np
#
# # 创建一个3*3的零矩阵，矩阵这里zeros函数的参数是一个tuple类型（3,3）
# data1 = np.mat(np.zeros((3,3)))
# print(data1)
# '''
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]
#
# '''
# # 创建一个2*4的1矩阵，默认是浮点型的数据，如果需要时int，可以使用dtype=int
# data2 = np.mat(np.ones((2,4)))
# print(data2)
# '''
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
# '''
#
# # 这里使用numpy的random模块
# # random.rand(2,2)创建的是一个二维数组，但是需要将其转化为matrix
# data3 = np.mat(np.random.rand(2,2))
# print(data3)
# '''
# [[0.62002668 0.55292404]
#  [0.53018371 0.1548954 ]]
# '''
#
# # 生成一个3*3的0-10之间的随机整数矩阵，如果需要指定下界可以多加一个参数
# data4  = np.mat(np.random.randint(10,size=(3,3)))
# print(data4)
# '''
# [[0 4 1]
#  [7 9 9]
#  [9 0 4]]
# '''
#
# # 产生一个2-8之间的随机整数矩阵
# data5 = np.mat(np.random.randint(2,8,size=(2,5)))
# print(data5)
# '''
# [[4 6 3 3 4]
#  [4 3 3 3 6]]
# '''
#
# # 产生一个2*2的对角矩阵
# data6 = np.mat(np.eye(2,2,dtype=int))
# print(data6)
# '''
# [[1 0]
#  [0 1]]
# '''
# # 生成一个对角线为1,2,3的对角矩阵
# a1 = [1,2,3]
# a2 = np.mat(np.diag(a1))
# print(a2)
# '''
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
# '''

# from numpy import  *
''' 1*2 的矩阵乘以2*1 的矩阵  得到1*1 的矩阵'''

# a1 = mat([1,2])
# print(a1)
# a2 = mat([[1],[2]])
# print(a2)
# a3 = a1*a2
# print(a3)
'''
[[1 2]]
[[1]
 [2]]
[[5]]
'''



# from numpy import  *
# ''' 矩阵点乘为对应矩阵元素相乘'''
#
# a1 = mat([1,1])
# print(a1)
# a2 = mat([2,2])
# print(a2)
# a3 = multiply(a1,a2)
# print(a3)
# '''
# [[1 1]]
# [[2 2]]
# [[2 2]]
# '''
#
#
# a1 = mat([2,2])
# a2 = a1*2
# print(a2)
# # [[4 4]]


# from numpy import *
# ''' 矩阵求逆变换:求矩阵matrix([[0.5,0],[0,0.5]])的逆矩阵'''
#
# a1 = mat(eye(2,2)*0.5)
# print(a1)
# a2 = a1.I
# print(a2)
# '''
# [[0.5 0. ]
#  [0.  0.5]]
# [[2. 0.]
#  [0. 2.]]
# '''

# from numpy import *
# '''矩阵的转置'''
#
# a1 = mat([[1,1],[0,0]])
# print(a1)
# a2 = a1.T
# print(a2)
# '''
# [[1 1]
#  [0 0]]
# [[1 0]
#  [1 0]]
#  '''

# from numpy import *
# '''计算每一列，行的和'''
#
# a1 = mat([[1,1],[2,3],[4,5]])
# print(a1)
# #  列和，这里得到的是1*2的矩阵
# a2=a1.sum(axis=0)
# print(a2)
# '''
# [[7 9]]
# '''
# #  行和，这里得到的是3*1的矩阵
# a3=a1.sum(axis=1)
# print(a3)
# '''
# [[2]
#  [5]
#  [9]]
#  '''
# #  计算第一行所有列的和，这里得到的是一个数值
# a4=sum(a1[1,:])
# print(a4)
# '''
# 5
# '''

# from numpy import *
# '''计算每一列，行的和'''
#
# a1 = mat([[1,1],[2,3],[4,5]])
# print(a1)
# '''
# [[1 1]
#  [2 3]
#  [4 5]]
# '''
# #  计算a1矩阵中所有元素的最大值,这里得到的结果是一个数值
# maxa = a1.max()
# print(maxa)   #5
# #  计算第二列的最大值，这里得到的是一个1*1的矩阵
# a2=max(a1[:,1])
# print(a2)   #[[5]]
# #  计算第二行的最大值，这里得到的是一个一个数值
# maxt = a1[1,:].max()
# print(maxt)   #3
# #   计算所有列的最大值，这里使用的是numpy中的max函数
# maxrow = np.max(a1,0)
# print(maxrow)   #[[4 5]]
# # ;//计算所有行的最大值，这里得到是一个矩阵
# maxcolumn = np.max(a1,1)
# print(maxcolumn)
# '''
# [[1]
#  [3]
#  [5]]
# '''
# #  计算所有列的最大值对应在该列中的索引
# maxindex = np.argmax(a1,0)
# print(maxindex)   #[[2 2]]
# #  计算第二行中最大值对应在改行的索引
# tmaxindex = np.argmax(a1[1,:])
# print(tmaxindex)  # 1


# from numpy import *
# ''' 矩阵的分隔，同列表和数组的分隔一致'''
#
# a = mat(ones((3,3)))
# print(a)
# # 分隔出第二行以后的行和第二列以后的列的所有元素
# b = a[1:,1:]
# print(b)
# '''
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]
# [[1. 1.]
#  [1. 1.]]
# '''


# from numpy import *
#
# a = mat(ones((2,2)))
# print(a)
# b = mat(eye(2))
# print(b)
# # 按照列和并，即增加行数
# c = vstack((a,b))
# print(c)
# # 按照行合并，即行数不变，扩展列数
# d = hstack((a,b))
# print(d)
# '''
# [[1. 1.]
#  [1. 1.]]
#
# [[1. 0.]
#  [0. 1.]]
#
# [[1. 1.]
#  [1. 1.]
#  [1. 0.]
#  [0. 1.]]
#
# [[1. 1. 1. 0.]
#  [1. 1. 0. 1.]]
# '''


# li =[[1],'hello',3]
# print(li)

# from numpy import *
# #
# a=array([[2],[1]])
# print(a )
# dimension=a.ndim
# m,n=a.shape
# # 元素总个数
# number=a.size
# print(number)
# # 2
# #  元素的类型
# str=a.dtype
# print(str)
# # int32


# from numpy import *
#
# # 列表
# a1 = [[1,2],[3,2],[5,2]]
# # 将列表转化为二维数组
# a2 = array(a1)
# # 将列表转化成矩阵
# a3 = mat(a1)
# # 将矩阵转化成数组
# a4 = array(a3)
# # 将矩阵转换成列表
# a5=a3.tolist()
# # 将数组转换成列表
# a6=a2.tolist()
# print(type(a1))
# print(type(a2))
# print(type(a3))
# print(type(a4))
# print(type(a5))
# print(type(a6))
# '''
# <class 'list'>
# <class 'numpy.ndarray'>
# <class 'numpy.matrixlib.defmatrix.matrix'>
# <class 'numpy.ndarray'>
# <class 'list'>
# <class 'list'>
# '''


# from numpy import *
#
# a1=[1,2,3]
# print(a1)
# print(type(a1))
# a2=array(a1)
# print(a2)
# print(type(a2))
# a3=mat(a1)
# print(a3)
# print(type(a3))
# '''
# [1, 2, 3]
# <class 'list'>
# [1 2 3]
# <class 'numpy.ndarray'>
# [[1 2 3]]
# <class 'numpy.matrixlib.defmatrix.matrix'>
#
# '''
# a4=a2.tolist()
# print(a4)
# print(type(a4))
# a5=a3.tolist()
# print(a5)
# print(type(a5))
# '''
# [1, 2, 3]
# <class 'list'>
# [[1, 2, 3]]
# <class 'list'>
# '''
#
# a6=(a4 == a5)
# print(a6)
# print(type(a6))
# a7=(a4 is a5[0])
# print(a7)
# print(type(a7))
# '''
# False
# <class 'bool'>
# False
# <class 'bool'>
# '''\


# dataMat=mat([1])
# print(dataMat)
# print(type(dataMat))
# '''
# [[1]]
# <class 'numpy.matrixlib.defmatrix.matrix'>
# '''
#
# # 这个时候获取的就是矩阵的元素的数值，而不再是矩阵的类型
# val=dataMat[0,0]
# print(val)
# print(type(val))
# '''
# 1
# <class 'numpy.int32'>
# '''


# li = [[1,1],[1,3],[2,3],[4,4],[2,4]]
# from numpy import *
#
# a = [[1,2],[3,4],[5,6]]
# a = mat(a)
# # 打印整个矩阵
# print(a[0:])
# '''
# [[1 2]
#  [3 4]
#  [5 6]]
#  '''
#
# # 打印矩阵E从1行开始到末尾行的内容
# print(a[1:])
# '''
# [[3 4]
#  [5 6]]
# '''
#
# # 表示打印矩阵E 从1行到3行的内容
# print(a[1:3])
# '''
# [[3 4]
#  [5 6]]
#  '''

li = [[1,1],[1,3],[2,3],[4,4],[2,4]]
from numpy import *
mat = mat(li)
# 在整个矩阵的基础下，打印1列（指的是序列为1的列
print(mat[:,0])
'''
[[1]
 [1]
 [2]
 [4]
 [2]]
'''
# 在矩阵的1行到2行（[1,3)） 的前提下打印两列
# 2 列不是指两列，而是序号为2的列
print(mat[1:3,1])
'''
[[3]
 [3]]
 '''
