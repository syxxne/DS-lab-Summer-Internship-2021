import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)

users = pd.read_csv("users.dat", sep="::", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="ISO-8859-1", engine="python")

# training-testing split
train, test = train_test_split(users, test_size=0.1, random_state=77)


ratings = pd.read_csv("ratings.dat", sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="ISO-8859-1", engine="python")

# 평점이 3 이상인 경우에만 1로 binarize
ratings.drop(ratings[ratings["Rating"] < 3].index, inplace=True)
ratings["Rating"] = 1

# 25개 이상 영화를 본 이용자만 추출
ratings = ratings.groupby("UserID").filter(lambda x: len(x) >= 25)


# 나이별로 user 분류
user_under_20 = train[train["Age"] < 20]
user_20s = train[(train["Age"] >= 20) & (train["Age"] <= 29)]
user_30s = train[(train["Age"] >= 30) & (train["Age"] <= 39)]
user_40s = train[(train["Age"] >= 40) & (train["Age"] <= 49)]
user_50s = train[(train["Age"] >= 50) & (train["Age"] <= 59)]

# 나이별 ratings dataframe 생성
user_under_20 = pd.merge(ratings, user_under_20, left_on="UserID", right_on="UserID", how="inner")
user_20s = pd.merge(ratings, user_20s, left_on="UserID", right_on="UserID", how="inner")
user_30s = pd.merge(ratings, user_30s, left_on="UserID", right_on="UserID", how="inner")
user_40s = pd.merge(ratings, user_40s, left_on="UserID", right_on="UserID", how="inner")
user_50s = pd.merge(ratings, user_50s, left_on="UserID", right_on="UserID", how="inner")

# 나이별 인기 있는 영화 ranking
rank_under_20 = user_under_20.groupby("MovieID").size().sort_values(ascending=False)
rank_under_20 = rank_under_20.values.tolist()
rank_20s = user_20s.groupby("MovieID").size().sort_values(ascending=False)
rank_20s = rank_20s.values.tolist()
rank_30s = user_30s.groupby("MovieID").size().sort_values(ascending=False)
rank_30s = rank_30s.values.tolist()
rank_40s = user_40s.groupby("MovieID").size().sort_values(ascending=False)
rank_40s = rank_40s.values.tolist()
rank_50s = user_50s.groupby("MovieID").size().sort_values(ascending=False)
rank_50s = rank_50s.values.tolist()

rank_under_20 = rank_under_20[0:15]
rank_20s = rank_20s[0:15]
rank_30s = rank_30s[0:15]
rank_40s = rank_40s[0:15]
rank_50s = rank_50s[0:15]


# test
test_under_20 = test[test["Age"] < 20]
test_20s = test[(test["Age"] >= 20) & (test["Age"] <= 29)]
test_30s = test[(test["Age"] >= 30) & (test["Age"] <= 39)]
test_40s = test[(test["Age"] >= 40) & (test["Age"] <= 49)]
test_50s = test[(test["Age"] >= 50) & (test["Age"] <= 59)]

test_under_20 = pd.merge(ratings, test_under_20, left_on="UserID", right_on="UserID", how="inner")
test_20s = pd.merge(ratings, test_20s, left_on="UserID", right_on="UserID", how="inner")
test_30s = pd.merge(ratings, test_30s, left_on="UserID", right_on="UserID", how="inner")
test_40s = pd.merge(ratings, test_40s, left_on="UserID", right_on="UserID", how="inner")
test_50s = pd.merge(ratings, test_50s, left_on="UserID", right_on="UserID", how="inner")

hit1 = 0

seen_under_20 = test_under_20.loc[:, "MovieID"].values.tolist()
seen_20s = test_20s.loc[:, "MovieID"].values.tolist()
seen_30s = test_30s.loc[:, "MovieID"].values.tolist()
seen_40s = test_40s.loc[:, "MovieID"].values.tolist()
seen_50s = test_50s.loc[:, "MovieID"].values.tolist()

for i in range(len(rank_under_20)):
    hit1 += seen_under_20.count(rank_under_20[i])

for i in range(len(rank_20s)):
    hit1 += seen_20s.count(rank_20s[i])

for i in range(len(rank_30s)):
    hit1 += seen_30s.count(rank_30s[i])

for i in range(len(rank_40s)):
    hit1 += seen_40s.count(rank_40s[i])

for i in range(len(rank_50s)):
    hit1 += seen_50s.count(rank_50s[i])


hit_rate_age = (hit1 / (len(seen_under_20) + len(seen_20s) + len(seen_30s) + len(seen_40s) + len(seen_50s))) * 100

print("Age Group hit rate : ", hit_rate_age)


# 직업별로 user 분류
oc0 = train[train["Occupation"] == 0]
oc1 = train[train["Occupation"] == 1]
oc2 = train[train["Occupation"] == 2]
oc3 = train[train["Occupation"] == 3]
oc4 = train[train["Occupation"] == 4]
oc5 = train[train["Occupation"] == 5]
oc6 = train[train["Occupation"] == 6]
oc7 = train[train["Occupation"] == 7]
oc8 = train[train["Occupation"] == 8]
oc9 = train[train["Occupation"] == 9]
oc10 = train[train["Occupation"] == 10]
oc11 = train[train["Occupation"] == 11]
oc12 = train[train["Occupation"] == 12]
oc13 = train[train["Occupation"] == 13]
oc14 = train[train["Occupation"] == 14]
oc15 = train[train["Occupation"] == 15]
oc16 = train[train["Occupation"] == 16]
oc17 = train[train["Occupation"] == 17]
oc18 = train[train["Occupation"] == 18]
oc19 = train[train["Occupation"] == 19]
oc20 = train[train["Occupation"] == 20]

# 직업별 ratings dataframe 생성
oc0 = pd.merge(ratings, oc0, left_on="UserID", right_on="UserID", how="inner")
oc1 = pd.merge(ratings, oc1, left_on="UserID", right_on="UserID", how="inner")
oc2 = pd.merge(ratings, oc2, left_on="UserID", right_on="UserID", how="inner")
oc3 = pd.merge(ratings, oc3, left_on="UserID", right_on="UserID", how="inner")
oc4 = pd.merge(ratings, oc4, left_on="UserID", right_on="UserID", how="inner")
oc5 = pd.merge(ratings, oc5, left_on="UserID", right_on="UserID", how="inner")
oc6 = pd.merge(ratings, oc6, left_on="UserID", right_on="UserID", how="inner")
oc7 = pd.merge(ratings, oc7, left_on="UserID", right_on="UserID", how="inner")
oc8 = pd.merge(ratings, oc8, left_on="UserID", right_on="UserID", how="inner")
oc9 = pd.merge(ratings, oc9, left_on="UserID", right_on="UserID", how="inner")
oc10 = pd.merge(ratings, oc10, left_on="UserID", right_on="UserID", how="inner")
oc11 = pd.merge(ratings, oc11, left_on="UserID", right_on="UserID", how="inner")
oc12 = pd.merge(ratings, oc12, left_on="UserID", right_on="UserID", how="inner")
oc13 = pd.merge(ratings, oc13, left_on="UserID", right_on="UserID", how="inner")
oc14 = pd.merge(ratings, oc14, left_on="UserID", right_on="UserID", how="inner")
oc15 = pd.merge(ratings, oc15, left_on="UserID", right_on="UserID", how="inner")
oc16 = pd.merge(ratings, oc16, left_on="UserID", right_on="UserID", how="inner")
oc17 = pd.merge(ratings, oc17, left_on="UserID", right_on="UserID", how="inner")
oc18 = pd.merge(ratings, oc18, left_on="UserID", right_on="UserID", how="inner")
oc19 = pd.merge(ratings, oc19, left_on="UserID", right_on="UserID", how="inner")
oc20 = pd.merge(ratings, oc20, left_on="UserID", right_on="UserID", how="inner")

# 직업별 인기 있는 영화 ranking
rank_oc0 = oc0.groupby("MovieID").size().sort_values(ascending=False)
rank_oc0 = rank_oc0.values.tolist()
rank_oc1 = oc1.groupby("MovieID").size().sort_values(ascending=False)
rank_oc1 = rank_oc1.values.tolist()
rank_oc2 = oc2.groupby("MovieID").size().sort_values(ascending=False)
rank_oc2 = rank_oc2.values.tolist()
rank_oc3 = oc3.groupby("MovieID").size().sort_values(ascending=False)
rank_oc3 = rank_oc3.values.tolist()
rank_oc4 = oc4.groupby("MovieID").size().sort_values(ascending=False)
rank_oc4 = rank_oc4.values.tolist()
rank_oc5 = oc5.groupby("MovieID").size().sort_values(ascending=False)
rank_oc5 = rank_oc5.values.tolist()
rank_oc6 = oc6.groupby("MovieID").size().sort_values(ascending=False)
rank_oc6 = rank_oc6.values.tolist()
rank_oc7 = oc7.groupby("MovieID").size().sort_values(ascending=False)
rank_oc7 = rank_oc7.values.tolist()
rank_oc8 = oc8.groupby("MovieID").size().sort_values(ascending=False)
rank_oc8 = rank_oc8.values.tolist()
rank_oc9 = oc9.groupby("MovieID").size().sort_values(ascending=False)
rank_oc9 = rank_oc9.values.tolist()
rank_oc10 = oc10.groupby("MovieID").size().sort_values(ascending=False)
rank_oc10 = rank_oc10.values.tolist()
rank_oc11 = oc11.groupby("MovieID").size().sort_values(ascending=False)
rank_oc11 = rank_oc11.values.tolist()
rank_oc12 = oc12.groupby("MovieID").size().sort_values(ascending=False)
rank_oc12 = rank_oc12.values.tolist()
rank_oc13 = oc13.groupby("MovieID").size().sort_values(ascending=False)
rank_oc13 = rank_oc13.values.tolist()
rank_oc14 = oc14.groupby("MovieID").size().sort_values(ascending=False)
rank_oc14 = rank_oc14.values.tolist()
rank_oc15 = oc15.groupby("MovieID").size().sort_values(ascending=False)
rank_oc15 = rank_oc15.values.tolist()
rank_oc16 = oc16.groupby("MovieID").size().sort_values(ascending=False)
rank_oc16 = rank_oc16.values.tolist()
rank_oc17 = oc17.groupby("MovieID").size().sort_values(ascending=False)
rank_oc17 = rank_oc17.values.tolist()
rank_oc18 = oc18.groupby("MovieID").size().sort_values(ascending=False)
rank_oc18 = rank_oc18.values.tolist()
rank_oc19 = oc19.groupby("MovieID").size().sort_values(ascending=False)
rank_oc19 = rank_oc19.values.tolist()
rank_oc20 = oc20.groupby("MovieID").size().sort_values(ascending=False)
rank_oc20 = rank_oc20.values.tolist()

rank_oc0 = rank_oc0[0:15]
rank_oc1 = rank_oc1[0:15]
rank_oc2 = rank_oc2[0:15]
rank_oc3 = rank_oc3[0:15]
rank_oc4 = rank_oc4[0:15]
rank_oc5 = rank_oc5[0:15]
rank_oc6 = rank_oc6[0:15]
rank_oc7 = rank_oc7[0:15]
rank_oc8 = rank_oc8[0:15]
rank_oc9 = rank_oc9[0:15]
rank_oc10 = rank_oc10[0:15]
rank_oc11 = rank_oc11[0:15]
rank_oc12 = rank_oc12[0:15]
rank_oc13 = rank_oc13[0:15]
rank_oc14 = rank_oc14[0:15]
rank_oc15 = rank_oc15[0:15]
rank_oc16 = rank_oc16[0:15]
rank_oc17 = rank_oc17[0:15]
rank_oc18 = rank_oc18[0:15]
rank_oc19 = rank_oc19[0:15]
rank_oc20 = rank_oc20[0:15]


# test
test_oc0 = test[test["Occupation"] == 0]
test_oc1 = test[test["Occupation"] == 1]
test_oc2 = test[test["Occupation"] == 2]
test_oc3 = test[test["Occupation"] == 3]
test_oc4 = test[test["Occupation"] == 4]
test_oc5 = test[test["Occupation"] == 5]
test_oc6 = test[test["Occupation"] == 6]
test_oc7 = test[test["Occupation"] == 7]
test_oc8 = test[test["Occupation"] == 8]
test_oc9 = test[test["Occupation"] == 9]
test_oc10 = test[test["Occupation"] == 10]
test_oc11 = test[test["Occupation"] == 11]
test_oc12 = test[test["Occupation"] == 12]
test_oc13 = test[test["Occupation"] == 13]
test_oc14 = test[test["Occupation"] == 14]
test_oc15 = test[test["Occupation"] == 15]
test_oc16 = test[test["Occupation"] == 16]
test_oc17 = test[test["Occupation"] == 17]
test_oc18 = test[test["Occupation"] == 18]
test_oc19 = test[test["Occupation"] == 19]
test_oc20 = test[test["Occupation"] == 20]

test_oc0 = pd.merge(ratings, test_oc0, left_on="UserID", right_on="UserID", how="inner")
test_oc1 = pd.merge(ratings, test_oc1, left_on="UserID", right_on="UserID", how="inner")
test_oc2 = pd.merge(ratings, test_oc2, left_on="UserID", right_on="UserID", how="inner")
test_oc3 = pd.merge(ratings, test_oc3, left_on="UserID", right_on="UserID", how="inner")
test_oc4 = pd.merge(ratings, test_oc4, left_on="UserID", right_on="UserID", how="inner")
test_oc5 = pd.merge(ratings, test_oc5, left_on="UserID", right_on="UserID", how="inner")
test_oc6 = pd.merge(ratings, test_oc6, left_on="UserID", right_on="UserID", how="inner")
test_oc7 = pd.merge(ratings, test_oc7, left_on="UserID", right_on="UserID", how="inner")
test_oc8 = pd.merge(ratings, test_oc8, left_on="UserID", right_on="UserID", how="inner")
test_oc9 = pd.merge(ratings, test_oc9, left_on="UserID", right_on="UserID", how="inner")
test_oc10 = pd.merge(ratings, test_oc10, left_on="UserID", right_on="UserID", how="inner")
test_oc11 = pd.merge(ratings, test_oc11, left_on="UserID", right_on="UserID", how="inner")
test_oc12 = pd.merge(ratings, test_oc12, left_on="UserID", right_on="UserID", how="inner")
test_oc13 = pd.merge(ratings, test_oc13, left_on="UserID", right_on="UserID", how="inner")
test_oc14 = pd.merge(ratings, test_oc14, left_on="UserID", right_on="UserID", how="inner")
test_oc15 = pd.merge(ratings, test_oc15, left_on="UserID", right_on="UserID", how="inner")
test_oc16 = pd.merge(ratings, test_oc16, left_on="UserID", right_on="UserID", how="inner")
test_oc17 = pd.merge(ratings, test_oc17, left_on="UserID", right_on="UserID", how="inner")
test_oc18 = pd.merge(ratings, test_oc18, left_on="UserID", right_on="UserID", how="inner")
test_oc19 = pd.merge(ratings, test_oc19, left_on="UserID", right_on="UserID", how="inner")
test_oc20 = pd.merge(ratings, test_oc20, left_on="UserID", right_on="UserID", how="inner")

hit2 = 0

seen_oc0 = test_oc0.loc[:, "MovieID"].values.tolist()
seen_oc1 = test_oc1.loc[:, "MovieID"].values.tolist()
seen_oc2 = test_oc2.loc[:, "MovieID"].values.tolist()
seen_oc3 = test_oc3.loc[:, "MovieID"].values.tolist()
seen_oc4 = test_oc4.loc[:, "MovieID"].values.tolist()
seen_oc5 = test_oc5.loc[:, "MovieID"].values.tolist()
seen_oc6 = test_oc6.loc[:, "MovieID"].values.tolist()
seen_oc7 = test_oc7.loc[:, "MovieID"].values.tolist()
seen_oc8 = test_oc8.loc[:, "MovieID"].values.tolist()
seen_oc9 = test_oc9.loc[:, "MovieID"].values.tolist()
seen_oc10 = test_oc10.loc[:, "MovieID"].values.tolist()
seen_oc11 = test_oc11.loc[:, "MovieID"].values.tolist()
seen_oc12 = test_oc12.loc[:, "MovieID"].values.tolist()
seen_oc13 = test_oc13.loc[:, "MovieID"].values.tolist()
seen_oc14 = test_oc14.loc[:, "MovieID"].values.tolist()
seen_oc15 = test_oc15.loc[:, "MovieID"].values.tolist()
seen_oc16 = test_oc16.loc[:, "MovieID"].values.tolist()
seen_oc17 = test_oc17.loc[:, "MovieID"].values.tolist()
seen_oc18 = test_oc18.loc[:, "MovieID"].values.tolist()
seen_oc19 = test_oc19.loc[:, "MovieID"].values.tolist()
seen_oc20 = test_oc20.loc[:, "MovieID"].values.tolist()

for i in range(len(rank_oc0)):
    hit2 += seen_oc0.count(rank_oc0[i])

for i in range(len(rank_oc1)):
    hit2 += seen_oc1.count(rank_oc1[i])

for i in range(len(rank_oc2)):
    hit2 += seen_oc2.count(rank_oc2[i])

for i in range(len(rank_oc3)):
    hit2 += seen_oc3.count(rank_oc3[i])

for i in range(len(rank_oc4)):
    hit2 += seen_oc4.count(rank_oc4[i])

for i in range(len(rank_oc5)):
    hit2 += seen_oc5.count(rank_oc5[i])

for i in range(len(rank_oc6)):
    hit2 += seen_oc6.count(rank_oc6[i])

for i in range(len(rank_oc7)):
    hit2 += seen_oc7.count(rank_oc7[i])

for i in range(len(rank_oc8)):
    hit2 += seen_oc8.count(rank_oc8[i])

for i in range(len(rank_oc9)):
    hit2 += seen_oc9.count(rank_oc9[i])

for i in range(len(rank_oc10)):
    hit2 += seen_oc10.count(rank_oc10[i])

for i in range(len(rank_oc11)):
    hit2 += seen_oc11.count(rank_oc11[i])

for i in range(len(rank_oc12)):
    hit2 += seen_oc12.count(rank_oc12[i])

for i in range(len(rank_oc13)):
    hit2 += seen_oc13.count(rank_oc13[i])

for i in range(len(rank_oc14)):
    hit2 += seen_oc14.count(rank_oc14[i])

for i in range(len(rank_oc15)):
    hit2 += seen_oc15.count(rank_oc15[i])

for i in range(len(rank_oc16)):
    hit2 += seen_oc16.count(rank_oc16[i])

for i in range(len(rank_oc17)):
    hit2 += seen_oc17.count(rank_oc17[i])

for i in range(len(rank_oc18)):
    hit2 += seen_oc18.count(rank_oc18[i])

for i in range(len(rank_oc19)):
    hit2 += seen_oc19.count(rank_oc19[i])

for i in range(len(rank_oc20)):
    hit2 += seen_oc20.count(rank_oc20[i])


hit_rate_oc = (hit2 / (len(seen_oc0) + len(seen_oc1) + len(seen_oc2) + len(seen_oc3) + len(seen_oc4) + len(seen_oc5) + len(seen_oc6) + len(seen_oc7) + len(seen_oc8) + len(seen_oc9) + len(seen_oc10) + len(seen_oc11) + len(seen_oc12) + len(seen_oc13) + len(seen_oc14) + len(seen_oc15) + len(seen_oc16) + len(seen_oc17) + len(seen_oc18) + len(seen_oc19) + len(seen_oc20))) * 100

print("Occupation Group hit rate : ", hit_rate_oc)