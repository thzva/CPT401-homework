import pandas as pd

users = pd.read_table('./ml-1m/users.dat',sep='::',engine='python',usecols=[3,4])
users.to_csv('./ml-1m/occ&Zip.txt')
