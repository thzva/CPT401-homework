import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def kmeans(data, k=3, normalize=False, limit=500):
    # normalize 数据
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]

    # 直接将前K个数据当成簇中心
    centers = data[:k]

    for i in range(limit):
        # 首先利用广播机制计算每个样本到簇中心的距离，之后根据最小距离重新归类
        classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
        # 对每个新的簇计算簇中心
        new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

        # 簇中心不再移动的话，结束循环
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
    else:
        # 如果在for循环里正常结束，下面不会执行
        raise RuntimeError(f"Clustering algorithm did not complete within {limit} iterations")

    # 如果normalize了数据，簇中心会反向 scaled 到原来大小
    if normalize:
        centers = centers * stats[1] + stats[0]

    return classifications, centers


data = pd.read_csv('./ml-1m/users.dat',
                      sep='::',  # 原文件的分隔符是'::'，此处是按此分隔符将数据导入
                      engine='python',
                      usecols=[2,3,4])
data = np.array(data)
classifications, centers = kmeans(data, normalize=True, k=20)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=classifications, s=100)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=500, c='k', marker='^');
plt.show()
# 8轮迭代

data2 = pd.read_csv('./ml-1m/users.dat',
                      sep='::',  # 原文件的分隔符是'::'，此处是按此分隔符将数据导入
                      engine='python',
                      usecols=[0,1,2,3,4],
                      header=None)
data2 = np.array(data2)

userid = data2[1:,0]
gender = data2[1:,1]
age = data2[1:, 2]
occupation = data2[1:, 3]
zipCode = data2[1:, 4]

data_to_save = []
for i in zip(userid, gender, age, occupation, zipCode, classifications):
    data_to_save.append([i[0], i[1], i[2], i[3], i[4], i[5]])
np_data = np.array(data_to_save)
##写入文件
pd_data = pd.DataFrame(np_data, columns=["UserID","Gender","Age","Occupation","Zip-code","Cluseterid"])
pd_data.to_csv('./ml-1m/newusers.dat', sep="\t", index=False)
pd_data = np.array(pd_data)
