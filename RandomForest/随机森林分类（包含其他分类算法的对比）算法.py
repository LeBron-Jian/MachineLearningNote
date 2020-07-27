#_*_coding:UTF_8_*_
 
# 导入需要导入的库
import pandas as pd
import  numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection ,metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
import warnings
 
# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")
 
# 每个样本有几个属性或者特征
n_features = 2
x,y = make_blobs(n_samples=300,n_features=n_features,centers=6)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.7)
 
# 绘制样本显示
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()
 
# 传统决策树，随机森林算法 极端随机数的区别
DT = DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
 
RF = RandomForestClassifier(n_estimators=10,max_features=math.sqrt(n_features),
                            max_depth=None,min_samples_split=2,bootstrap=True)
 
EC = ExtraTreesClassifier(n_estimators=10,max_features=math.sqrt(n_features),
                          max_depth=None,min_samples_split=2,bootstrap=False)
 
 
 
# 训练
DT.fit(x_train,y_train)
RF.fit(x_train,y_train)
EC.fit(x_train,y_train)
 
#区域预测
# 第0列的范围
x1_min,x1_max = x[:,0].min(),x[:,0].max()
# 第1列的范围
x2_min,x2_max = x[:,1].min(),x[:,1].max()
# 生成网格采样点行列均为200点
x1,x2 = np.mgrid[x1_min:x1_max:200j,x2_min:x2_max:200j]
# 将区域划分为一系列测试点用去学习的模型预测，进而根据预测结果画区域
area_sample_point = np.stack((x1.flat,x2.flat),axis=1)
 
 
# 所有区域点进行预测
area1_predict = DT.predict(area_sample_point)
area1_predict = area1_predict.reshape(x1.shape)
 
 
area2_predict = RF.predict(area_sample_point)
area2_predict = area2_predict.reshape(x1.shape)
 
area3_predict = EC.predict(area_sample_point)
area3_predict = area3_predict.reshape(x1.shape)
 
 
# 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = [u'SimHei']
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
 
# 区域颜色
classifier_area_color = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
# 样本所属类别颜色
cm_dark = mpl.colors.ListedColormap(['r','g','b'])
 
# 绘图
# 第一个子图
plt.subplot(2,2,1)
 
plt.pcolormesh(x1,x2,area1_predict,cmap = classifier_area_color)
plt.scatter(x_train[:,0],x_train[:,1],c =y_train,marker='o',s=50,cmap=cm_dark)
plt.scatter(x_test[:,0],x_test[:,1],c =y_test,marker='x',s=50,cmap=cm_dark)
 
plt.xlabel('data_x',fontsize=8)
plt.ylabel('data_y',fontsize=8)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title(u'DecisionTreeClassifier: 传统决策树',fontsize=8)
plt.text(x1_max-9,x2_max-2,u'o-------train ; x--------test$')
 
# 第二个子图
plt.subplot(2,2,2)
 
plt.pcolormesh(x1,x2,area2_predict,cmap = classifier_area_color)
plt.scatter(x_train[:,0],x_train[:,1],c =y_train,marker='o',s=50,cmap=cm_dark)
plt.scatter(x_test[:,0],x_test[:,1],c =y_test,marker='x',s=50,cmap=cm_dark)
 
plt.xlabel('data_x',fontsize=8)
plt.ylabel('data_y',fontsize=8)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title(u'RandomForestClassifier: 随机森林算法',fontsize=8)
plt.text(x1_max-9,x2_max-2,u'o-------train ; x--------test$')
 
# 第三个子图
plt.subplot(2,2,3)
 
plt.pcolormesh(x1,x2,area3_predict,cmap = classifier_area_color)
plt.scatter(x_train[:,0],x_train[:,1],c =y_train,marker='o',s=50,cmap=cm_dark)
plt.scatter(x_test[:,0],x_test[:,1],c =y_test,marker='x',s=50,cmap=cm_dark)
 
plt.xlabel('data_x',fontsize=8)
plt.ylabel('data_y',fontsize=8)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title(u'ExtraTreesClassifier: 极端随机树',fontsize=8)
plt.text(x1_max-9,x2_max-2,u'o-------train ; x--------test$')
 
# 第四个子图
plt.subplot(2,2,4)
y = []
# 交叉验证
score_DT = cross_val_score(DT,x_train,y_train)
y.append(score_DT.mean())
score_RF = cross_val_score(RF,x_train,y_train)
y.append(score_RF.mean())
score_EC = cross_val_score(EC,x_train,y_train)
y.append(score_EC.mean())
 
print('DecisionTreeClassifier交叉验证准确率为:'+str(score_DT.mean()))
print('RandomForestClassifier交叉验证准确率为:'+str(score_RF.mean()))
print('ExtraTreesClassifier交叉验证准确率为:'+str(score_EC.mean()))
 
x = [0,1,2]
plt.bar(x,y,0.4,color='green')
plt.xlabel("0--DecisionTreeClassifier;1--RandomForestClassifier;2--ExtraTreesClassifie", fontsize=8)
plt.ylabel("平均准确率", fontsize=8)
plt.ylim(0.9, 0.99)
plt.title("交叉验证", fontsize=8)
for a, b in zip(x, y):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 
plt.show()