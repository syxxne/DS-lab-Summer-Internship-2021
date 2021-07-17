import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)


users = pd.read_csv("users.dat", sep="::", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="ISO-8859-1", engine="python")

# training-testing split
train, test = train_test_split(users, test_size=0.1, random_state=77)


ratings = pd.read_csv("ratings.dat", sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="ISO-8859-1", engine="python")

# 25개 이상 영화를 본 이용자만 추출
ratings = ratings.groupby("UserID").filter(lambda x: len(x) >= 25)

# 추출된 데이터들 중, train과 test set에 대한 ratings dataframe 생성
train_ratings = pd.merge(ratings, train, left_on="UserID", right_on="UserID", how="inner")
test_ratings = pd.merge(ratings, test, left_on="UserID", right_on="UserID", how="inner")

seen_movies = train_ratings.loc[:, "MovieID"]
movie_ID, movie_counts = np.unique(seen_movies, return_counts=True)

# 전체 영화 랭킹
rankings = dict(zip(movie_ID, movie_counts))
rankings = sorted(rankings.items(), reverse=True, key=lambda x: x[1])

rankings_df = pd.DataFrame(rankings, columns = ["MovieID", "Counts"])


movies = pd.read_csv("movies.dat", sep="::", names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1", engine="python")

movies["Genres"] = movies["Genres"].str.split("|")

# movie와 raking 데이터프레임 join하여 랭킹에 따라 movie 특성 정렬
movies = pd.merge(movies, rankings_df, left_on="MovieID", right_on="MovieID", how="inner")
movies = movies.sort_values("Counts", ascending=False)

movies["Genres_string"] = movies["Genres"].apply(lambda x : (" ").join(x))

# 영화 장르별 랭킹
Action = movies[movies["Genres_string"].str.contains("Action", case=False)]

# user와 movie 데이터프레임 join하여 이용자와 해당 이용자가 본 영화 장르 연관 짓기
user_genres = pd.merge(test_ratings, movies, left_on="MovieID", right_on="MovieID", how="inner")
user_genres = user_genres.loc[:, ["UserID", "MovieID", "Genres"]]

# 이용자별 favorite 장르(가장 많이 본 장르) 찾기
fav_genre = user_genres.groupby("UserID")["Genres"].agg(**{"FavGenre" : lambda x:x.mode().values[0]}).reset_index()
fav_genre["FavGenre"] = fav_genre["FavGenre"].apply(lambda x:Counter(x).most_common(1)[0][0])

test_userID = fav_genre.loc[:, "UserID"].values.tolist()

hit = 0

# hit value 구하기
for i in range(len(test_userID)):
    rec_movie = movies[movies["Genres_string"].str.contains(fav_genre.iloc[i][1], case=False)]
    rec_movie = rec_movie.loc[:, "MovieID"]
    rec_movie_list = rec_movie.values.tolist()

    if len(rec_movie_list) > 20:
        rec_movie_list = rec_movie_list[0:20]

    seen_df = test_ratings.loc[test_ratings["UserID"] == test_userID[i], "MovieID"]
    seen_list = seen_df.values.tolist()

    for j in range(len(rec_movie_list)):
        hit += seen_list.count(rec_movie_list[j])

hit_rate = (hit / len(test_ratings.index)) * 100



