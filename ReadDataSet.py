import pandas as pd
rnames = ["UserID", "MovieID", "Rating", "TimeStamp"]
ratings = pd.read_table("./ml-1m/ratings.dat", sep="::", header=None, names=rnames, engine='python')
print(ratings[:5])

unames = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
users = pd.read_table("./ml-1m/users.dat", sep="::", header=None, names=unames, engine='python')
print(users[:5])

mnames = ["MovieID", "Title", "Genres"]
movies = pd.read_table("./ml-1m/movies.dat", sep="::", header=None, names=mnames, engine='python')
print(movies[:5])