import pandas as pd
from math import pow, sqrt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

u_cols = ['user_id', 'age', 'sex']
users = pd.read_csv('D:/retour/recommendation/user.csv', encoding='latin-1')
print(users)

# Reading ratings dataset into a pandas dataframe object.
r_cols = ['user_id', 'product_id', 'rating']
ratings = pd.read_csv('D:/retour/recommendation/ratings.csv', encoding='latin-1')
print(ratings.head())

# Reading products dataset into a pandas dataframe object.
p_cols = ['product_id', 'product_name', 'category']
products = pd.read_csv('D:/retour/recommendation/products.csv', encoding='latin-1')
print(products)

# Getting distinct category types for generating columns of category type.

category_columns=products.category.unique().tolist()

print (category_columns)
# Iterating over every list to create and fill values into columns.
a=products.copy()
a=pd.get_dummies(a['category'])
products=pd.concat([products, a], axis=1)
products.head()

print(products.head())
print(ratings.to_string())
# Function to get the rating given by a user to a product.
def get_rating_(userid, productid):
    return ratings.loc[(ratings.user_id == userid) & (ratings.product_id == productid), 'rating'].iloc[0]


# Function to get the list of all product ids the specified user has rated.
def get_productids_(userid):
    return ratings.loc[(ratings.user_id == userid), 'product_id'].tolist()


# Function to get the product names against the product id.
def get_product_name_(productid):
    return products.loc[(products.product_id == productid), 'product_name'].iloc[0]


def distance_similarity_score(user1, user2):
    """
    distance between the two users (i.e. Euclidean distances)
    """
    both_buy_count = 0
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            both_buy_count += 1
    if both_buy_count == 0:
        return 0
    distance = []
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            rating1 = get_rating_(user1, element)
            rating2 = get_rating_(user2, element)
            distance.append(pow(rating1 - rating2, 2))
    total_distance = sum(distance)
    return 1 / (1 + sqrt(total_distance))


distance_similarity_score(1, 5)


def pearson_correlation_score(user1, user2):
    """
    user1 & user2 : user ids of two users between which similarity score is to be calculated.
    """
    both_buy_count = []
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            both_buy_count.append(element)
    if len(both_buy_count) == 0:
        return 0
    rating_sum_1 = sum([get_rating_(user1, element) for element in both_buy_count])
    rating_sum_2 = sum([get_rating_(user2, element) for element in both_buy_count])
    rating_squared_sum_1 = sum([pow(get_rating_(user1, element), 2) for element in both_buy_count])
    rating_squared_sum_2 = sum([pow(get_rating_(user2, element), 2) for element in both_buy_count])
    product_sum_rating = sum([get_rating_(user1, element) * get_rating_(user2, element) for element in both_buy_count])

    numerator = product_sum_rating - ((rating_sum_1 * rating_sum_2) / len(both_buy_count))
    denominator = sqrt((rating_squared_sum_1 - pow(rating_sum_1, 2) / len(both_buy_count)) * (
            rating_squared_sum_2 - pow(rating_sum_2, 2) / len(both_buy_count)))
    if denominator == 0:
        return 0
    return numerator / denominator


pearson_correlation_score(1, 5)


# Most similar users using two scores i.e., Euclidean distance and pearson correlation.
def most_similar_users_(user1, number_of_users, metric='pearson'):
    # Getting distinct user ids.
    user_ids = ratings.user_id.unique().tolist()

    # Getting similarity score between targeted and every other user in the list(or subset of the list).
    if metric == 'pearson':
        similarity_score = [(pearson_correlation_score(user1, nth_user), nth_user) for nth_user in user_ids[:30] if
                            nth_user != user1]
    else:
        similarity_score = [(distance_similarity_score(user1, nth_user), nth_user) for nth_user in user_ids[:30] if
                            nth_user != user1]

    # Sorting in descending order.
    similarity_score.sort()
    similarity_score.reverse()

    # Returning the top most 'number_of_users' similar users.
    return similarity_score[:number_of_users]


# Getting product Recommendations for Targeted User

def get_recommendation_(userid):
    user_ids = ratings.user_id.unique().tolist()
    total = {}
    similariy_sum = {}

    # Iterating over subset of user ids.
    for user in user_ids[:30]:

        # not comparing the user to itself
        if user == userid:
            continue

        # Getting similarity score between the users.
        score = pearson_correlation_score(userid, user)

        # not considering users having zero or less similarity score.
        if score <= 0:
            continue

        # Getting weighted similarity score and sum of similarities between both the users.
        for productid in get_productids_(user):
            # Only considering not bought/not rated products
            if productid not in get_productids_(userid) or get_rating_(userid, productid) == 0:
                total[productid] = 0
                total[productid] += get_rating_(user, productid) * score
                similariy_sum[productid] = 0
                similariy_sum[productid] += score

    # Normalizing ratings
    ranking = [(tot / similariy_sum[productid], productid) for productid, tot in total.items()]
    ranking.sort()
    ranking.reverse()

    # Getting product names against the product ids.
    recommendations = [get_product_name_(productid) for score, productid in ranking]
    return recommendations[:10]


print(most_similar_users_(5, 6))

print(get_recommendation_(4))

ratedproducts = pd.merge(ratings, products, on='product_id')

if not ratedproducts[ratedproducts.duplicated(['user_id', 'product_name'])].empty:
    new_ratings = ratedproducts.drop_duplicates(['user_id', 'product_name'])
df_for_knn = new_ratings.pivot(index='user_id', columns='product_name', values='rating').fillna(0)
print(df_for_knn.head())
df_for_knn_sparse = csr_matrix(df_for_knn.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(df_for_knn_sparse)



userid = 5
n_neighbors = 6  # for top 5 (the first is the user itself)
distances, indices = model_knn.kneighbors(df_for_knn.loc[userid].values.reshape(1, -1),
                                          n_neighbors=6)  # get the top k similar users to this particular user
print("Nearest Neighbors of User with ID = {0}:".format(userid))
for i in range(0, len(distances.flatten())):
    if i != 0:  # avoid the first user that is the user itself
        print("{0}: UserId: {1} -- Distance: {2}".format(i, df_for_knn.index[indices.flatten()[i]],
                                                         distances.flatten()[i]))

sum = 0
weightsum = 0
product_name = 'FruitCake'  # the product we'll try to predict the rating
for i in range(0, len(distances.flatten())):
    if i != 0:
        indice = df_for_knn.index[indices.flatten()[i]]
        weight = 1 - distances.flatten()[i]  # computes the weights according to the similarity
        sum = sum + df_for_knn.loc[indice, product_name] * weight  # computes the weighted sum
        weightsum = weightsum + weight
print("The rating that the user with ID = {0} will give in the product {1} is: {2}".format(userid, product_name,
                                                                                           round(sum / weightsum, 1)))

for j in range(0, len(distances.flatten())):
    if j != 0:
        indice = df_for_knn.index[indices.flatten()[j]]

print("The rating that the product with name = {0} will better be recommended by customer id {1}: ".format(product_name,
                                                                                                           indice))
