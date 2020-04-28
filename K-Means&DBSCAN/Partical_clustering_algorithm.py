# beer dataset from my github
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def load_data(filename, scaled_Flags=False):
    beer = pd.read_csv(filename, sep=' ')
    # print(beer)
    # 提取变量
    features = beer[['calories', 'sodium', 'alcohol', 'cost']]
    # print(X)
    if scaled_Flags == True:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    # print(features)
    return beer, features


def K_means_Clustering(beer, features):
    km1 = KMeans(n_clusters=3)
    clf1 = km1.fit(features)
    km2 = KMeans(n_clusters=2)
    clf2 = km2.fit(features)
    # print(clf1.labels_)
    # print(clf2.labels_)
    beer['cluster1'] = clf1.labels_
    beer['cluster2'] = clf2.labels_
    beer['scaled_cluster'] = clf1.labels_
    sort_res = beer.sort_values('scaled_cluster')
    # print(sort_res)
    cluster_centers1 = clf1.cluster_centers_
    cluster_centers2 = clf2.cluster_centers_
    # print(cluster_centers1)
    # print(cluster_centers2)
    group_res1 = beer.groupby('cluster1').mean()
    group_res2 = beer.groupby('cluster2').mean()
    group_res3 = beer.groupby('scaled_cluster').mean()
    # print(group_res1)
    # print(group_res2)
    # print(group_res3)
    # plt.rcParams['font.size'] = 14
    # colors = np.array(['red', 'green', 'blue', 'yellow'])
    # scatter_matrix(beer[['calories', 'sodium', 'alcohol', 'cost']],
    #                s=100, alpha=1, c=colors[beer['cluster2']], figsize=(10, 10))
    # plt.title('With 3 centroids initialized')

    # score_scaled = metrics.silhouette_score(features, beer.scaled_cluster)
    # score1 = metrics.silhouette_score(features, beer.cluster1)
    # score2 = metrics.silhouette_score(features, beer.cluster2)
    # print(score_scaled, score1, score2)
    scores = []
    for k in range(2, 20):
        clf = KMeans(n_clusters=k).fit(features)
        lables = clf.labels_
        score = metrics.silhouette_score(features, lables)
        scores.append(score)
    # print(scores)
    plt.plot(list(range(2, 20)), scores)
    plt.xlabel('Number of Clusters Initialized')
    plt.ylabel('Sihouette Score')

    return group_res1, group_res2


def draw_graph(group_res1, group_res2):
    centers1 = group_res1.reset_index()
    plt.rcParams['font.size'] = 14
    colors = np.array(['red', 'green', 'blue', 'yellow'])
    plt.scatter(beer['calories'], beer['alcohol'], c=colors[beer['cluster1']])
    plt.scatter(centers1.calories, centers1.alcohol, linewidths=3, marker='+', s=300, c='black')
    plt.xlabel('Calories')
    plt.ylabel('Alcohol')
    scatter_matrix(beer[['calories', 'sodium', 'alcohol', 'cost']],
                   s=100, alpha=1, c=colors[beer['scaled_cluster']], figsize=(10, 10))
    # scatter_matrix(features,
    #                s=100, alpha=1, c=colors[beer.scaled_cluster], figsize=(10, 10))
    # plt.title('With 3 centroids initialized')


def DBSCAN_Clustering(beer, features):
    db = DBSCAN(eps=10, min_samples=2).fit(features)
    labels = db.labels_
    beer['cluster_db'] = labels
    sort_res = beer.sort_values('cluster_db')
    # print(sort_res)
    group_res = beer.groupby('cluster_db').mean()
    # print(group_res)
    # plt.rcParams['font.size'] = 14
    # colors = np.array(['red', 'green', 'blue', 'yellow'])
    # scatter_matrix(beer[['calories', 'sodium', 'alcohol', 'cost']],
    #                s=100, alpha=1, c=colors[beer['cluster_db']], figsize=(10, 10))

    # score_scaled = metrics.silhouette_score(features, beer.scaled_cluster)
    # score1 = metrics.silhouette_score(features, beer.cluster_db)
    # print(score1)
    scores = []
    for k in range(5, 10, 1):
        clf = DBSCAN(eps=10, min_samples=k).fit(features)
        lables = clf.labels_
        score = metrics.silhouette_score(features, lables)
        scores.append(score)
    print(scores)


if __name__ == '__main__':
    filename = 'data.txt'
    beer, features = load_data(filename, True)
    group_res1, group_res2 = K_means_Clustering(beer, features)
    # draw_graph(group_res1, group_res2)
    # DBSCAN_Clustering(beer, features)
