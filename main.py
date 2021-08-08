import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_row", None)

ratings = pd.read_csv("ratings.dat", sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="ISO-8859-1", engine="python")
ratings.drop(["Timestamp"], axis=1, inplace=True)

# 모든 평점을 1로 binarize
ratings["Rating"] = 1

# training-testing split
x = ratings.copy()
train, test = train_test_split(x, test_size=0.1, random_state=77)


movies = pd.read_csv("movies.dat", sep="::", names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1", engine="python")
movies.drop(["Genres"], axis=1, inplace=True)

movie_ratings = pd.merge(ratings, movies, on= "MovieID")


matrix = movie_ratings.pivot(values="Rating", index="UserID", columns="MovieID").fillna(0)
U, sigma, Vt = svds(matrix, k=13)

sigma = np.diag(sigma)

svd_pred_ratings = np.dot(np.dot(U, sigma), Vt)
pred_ratings = pd.DataFrame(svd_pred_ratings, columns=matrix.columns)


count = 0

for i in range(len(test.index)):
    idx1 = test.iloc[i][0] - 1
    idx2 = test.iloc[i][1]
    if (pred_ratings.iloc[idx1][idx2] >= 0.3) :
        count += 1

print("0.3 기준 :", (count / len(test)) * 100, "%")


count = 0

for i in range(len(test.index)):
    idx1 = test.iloc[i][0] - 1
    idx2 = test.iloc[i][1]
    if (pred_ratings.iloc[idx1][idx2] >= 0.5):
        count += 1

print("0.5 기준 :", (count / len(test)) * 100, "%")


count = 0

for i in range(len(test.index)):
    idx1 = test.iloc[i][0] - 1
    idx2 = test.iloc[i][1]
    if (pred_ratings.iloc[idx1][idx2] >= np.mean(pred_ratings.iloc[idx1])):
        count += 1

print("평균 기준 :", (count / len(test)) * 100, "%")


count = 0

for i in range(len(test.index)):
    idx1 = test.iloc[i][0] - 1
    idx2 = test.iloc[i][1]
    if (pred_ratings.iloc[idx1][idx2] >= np.median(pred_ratings.iloc[idx1])):
        count += 1

print("중앙값 기준 :", (count / len(test)) * 100, "%")
