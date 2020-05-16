from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

filename = 'Fremont.csv'
data = pd.read_csv(filename, index_col='Date', parse_dates=True)
res = data.head()
# print(data.shape)
# 展示原数据
# data.plot()

# 对数据进行重采样，按照一周来看
# data.resample('w').sum().plot()

# 对数据采用滑动窗口，这里窗口是365天一个
# data.resample('D').sum().rolling(365).sum().plot()

# 查看某一天的数据分布
# data.groupby(data.index.time).mean().plot()
# plt.xticks(rotation=45)

# 查看前五个小时时间变换
# pivot table
data.columns = ['West', 'East']
data['Total'] = data['West'] + data['East']
pivoted = data.pivot_table('Total', index=data.index.time, columns=data.index.date)
res = pivoted.iloc[:5, :5]
# print(res)
# print(pivoted.shape)
# 画图展示一下
# pivoted.plot(legend=False, alpha=0.01)
# plt.xticks(rotation=45)

X = pivoted.fillna(0).T.values
print(X.shape)
X2 = PCA(2).fit_transform(X)
print(X2.shape)
# plt.scatter(X2[:, 0], X2[:, 1])
#

gmm = GaussianMixture(2)
gmm.fit(X)
labels_prob = gmm.predict_proba(X)
print(labels_prob)
labels = gmm.predict(X)
print(labels)
# plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='rainbow')


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
pivoted.T[labels == 0].T.plot(legend=False, alpha=0.1, ax=ax[0])
pivoted.T[labels == 1].T.plot(legend=False, alpha=0.1, ax=ax[1])
ax[0].set_title('Purple Cluster')
ax[1].set_title('Red Cluster')
