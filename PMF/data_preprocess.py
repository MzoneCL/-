import pandas as pd
import os
import numpy as np
import pickle

'''

    数据预处理

'''

ratings_path = 'row_data/ratings.csv'
movies_path = 'row_data/movies.csv'

ratings = pd.read_csv(ratings_path)
# print(ratings.describe())

user_Ids = list(set(ratings['userId']))
user_Id_index = dict((user_id, index) for user_id, index in zip(user_Ids, range(len(user_Ids))))
movie_Ids = list(set(ratings['movieId']))
movie_Id_index = dict((item_id, index) for item_id, index in zip(movie_Ids, range(len(movie_Ids))))

# print('userIds length: ' + str(user_Ids.__len__()))
# print('movieIds length: ' + str(movie_Ids.__len__()))
# print(movie_Id_index)

data = []
for i in range(len(ratings)):
    r = ratings.iloc[i]    
    
    user_id = int(r.userId)
    movie_id = int(r.movieId)
    rating = float(r.rating)

    data.append([user_Id_index[user_id], movie_Id_index[movie_id], rating])
    

data = np.array(data)
np.random.shuffle(data)

pickle.dump(user_Id_index, open(os.path.join('processed_data', 'user_id_index.pkl'), 'wb'))
pickle.dump(movie_Id_index, open(os.path.join('processed_data', 'movie_id_index.pkl'), 'wb'))
np.savetxt(os.path.join('processed_data', 'data.txt'), data, fmt='%d %d %0.1f')

