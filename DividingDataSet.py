import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("./ml-1m/newrating.dat")
train,test = train_test_split(data,test_size=0.2,)  #测试集的比例是20%
np.savetxt("./ml-1m/ml_train.csv",train,delimiter=",",fmt='%s') #delimiter=“，” 才能一个数据一个格
np.savetxt("./ml-1m/ml_test.csv",test,delimiter=",",fmt='%s')
