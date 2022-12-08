import pandas as pd

#--------------------------去除邮编中的地址后缀-------------------------------

#读取文件
users = pd.read_csv('./ml-1m/users.dat',
                      sep='::',
                      engine='python',
                      usecols=[4])

#去除有编中所有‘-’和后面的数字
# def removeline(x):
#    key = '-'
#    if key in x:
#      key_index = x.index(key)
#      x = x.replace(x,x[:key_index])
#    return x

#遍历第一列所有行
# for i in range(len(users)):
#     x = users.iat[i, 0]
#     users.replace(lambda m: x,removeline(x), inplace=True)

#查看效果
# for i in range(len(users)):
#    print(users.iat[i, 0])

#--------------------------去除邮编中过大的值-------------------------------
# def removeBigData(x):
#     t =int(x)
#     if t > 99999:
#       x = x.replace(x,x[:4])
#     return x
#
# for i in range(len(users)):
#     x = users.iat[i, 0]
#     users.replace(x,(lambda x:removeBigData(x)))

temp = -100
index = 0
for i in range(len(users)):
    x = users.iat[i, 0]
    if x > temp:
        temp = x
        index = i
        print(temp)
print(index+2)
