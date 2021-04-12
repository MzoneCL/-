import numpy as np
import pandas as pd


def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat[row,col]=values
    
    return mat




# 索引 到 movieId 的转换
def idx2movieId(movies):
    mvId2Idx = movieId2idx(movies=movies)
    return {value:key for key, value in mvId2Idx.items()}



# movieId 到 索引 的转换
# 传入的参数是 数据集：movies.csv
def movieId2idx(movies):
    movieIds = movies.movieId
    mvId2Idx = {}
    for (idx, mvId) in enumerate(movieIds):
        mvId2Idx[mvId] = idx

    return mvId2Idx    


# 获取数据集，返回值为 DataFrame，只含有 userId movieIdx rating 三列
def get_data(ratings, movies):
    movies['movieIdx'] = movies.index # 如果按照 movies 原本的 mivieId 组成 mat，会导致矩阵过大，浪费内存，所以，用索引代替
    merge_inner = pd.merge(ratings, movies, on='movieId') # 按 电影id 合并两个df
    merge_inner = merge_inner[['userId', 'movieIdx', 'rating']] # 只取我们需要的属性

    return merge_inner


# 通过 ratings 和 movies 两个 dataframe 得到评分矩阵
def df2mat(ratings, movies, num_users, num_items):
    merge_inner = get_data(ratings=ratings,movies=movies)

    df_array =  np.array(merge_inner)
    mat = np.zeros([num_users, num_items])
    row_idx = df_array[:, 0].astype(int)
    col_idx = df_array[:, 1].astype(int)
    
    ratings = df_array[:, 2].astype(np.float32)
    mat[row_idx, col_idx] = ratings

    return mat
