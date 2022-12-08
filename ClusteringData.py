import numpy as np
import pandas as pd

info1 = pd.read_csv('./ml-1m/newusers.dat',
                    sep='\t',  # 原文件的分隔符是'::'，此处是按此分隔符将数据导入
                    engine='python',
                    usecols=[0, 5],
                    header=None)

info2 = pd.read_csv('./ml-1m/ratings.dat',
                    sep='::',  # 原文件的分隔符是'::'，此处是按此分隔符将数据导入
                    engine='python',
                    usecols=[0, 1, 2, 3],
                    header=None)

data1 = np.array(info1)[1:]  # 不要第一行header数据
userid = np.array(np.where(data1[:, 1] == '10'))  # 聚类编号为10的读进来


key = np.array(info2)[:, 0]         # 将用户ID作为对比的关键词
data2 = np.array(info2)
total = 1000209                     # data2总共这么多组数据

temp = []
i =0
for i in range(total):
    if (data2[i,0]-1) in userid:            #因为传入的userid比实际值小1
        temp.append(data2[i,:])
temp = np.array(temp)
pd_data = pd.DataFrame(temp, columns=["UserID", "MovieID", "Rating", "TimeStamp"])
pd_data.to_csv('./ml-1m/newrating.dat', index=False)
pd_data = np.array(pd_data)