from math import sqrt
import numpy as np
from numpy import genfromtxt
import sys
import itertools as itr
from operator import itemgetter
import copy

class User():
    def __init__(self, id, items):
        self.id = id
        self.items = items

def loadData():
    data = genfromtxt('small-dataset.csv', delimiter=',',dtype="U")

    users = []
    for user in data:
        users.append(User(str(user[:1][0]), user[1:].astype(float)))

    return users

def user_sim_cosine_sim(person1, person2):
# computes similarity between two users based on the cosine similarity metric
    pass

def user_sim_pearson_corr(person1, person2):
# computes similarity between two users based on the pearson similarity metric
    pass

def most_similar_users(person, number_of_users):
    pass
# returns top-K similar users for the given

def user_recommendations(user, similar_data):

    pass
# generate recommendations for the given user

def userbased_cf(users):
    combs = itr.combinations(users, 2)
    combs_c = list(copy.deepcopy(combs))

    user_similar = [] # Pair of users
    for c in combs:
        items1 = c[0].items
        items2 = c[1].items
        items = np.vstack((items1, items2))
        items = items.transpose() # Temporarily transposes

        non_zero_items = items[np.all(items != 0, axis=1)].transpose()

        diff = non_zero_items[0] - non_zero_items[1]
        avg = sum(diff) / len(diff)

        user_similar.append((c[0].id, c[1].id, avg))


    return user_similar, combs_c

def best_usermatch(user, similar_data):
    user_similar = [x for x in similar_data if user.id in x]

    best_usermatch_data = list(max(user_similar,key=itemgetter(2)))
    best_usermatch = list(filter(lambda x: isinstance(x, str), best_usermatch_data))
    best_usermatch.remove(user.id)

    return best_usermatch[0]


def best_recommendation_for_user(user_id, best_usermatch, combs):
    user = None
    best_user = None
    for c in combs:
        if c[0].id == best_usermatch and c[1].id == user_id or c[1].id == best_usermatch and c[0].id == user_id:
            if c[0].id == user_id:
                user = c[0]
                best_user = c[1]
            else:
                user = c[1]
                best_user = c[0]

    user_not_seen = []
    for i in range(len(user.items)):
        if user.items[i] == 0:
            user_not_seen.append((i, best_user.items[i]))

    recomended_item = list(max(user_not_seen,key=itemgetter(1)))
    print("The recomended item for {:s} is number {:d} which was rated {:f} by user {:s}".format(user.id, recomended_item[0], recomended_item[1], best_user.id))


#################################### START ####################################

users = loadData()

user_similar, combs = userbased_cf(users)


user_to_check = users[2]
best_usermatch = best_usermatch(user_to_check, user_similar)

best_recommendation_for_user(user_to_check.id, best_usermatch, combs)
