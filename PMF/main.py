import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle
from pmf import PMF
from evaluation import get_rmse

if __name__ == '__main__':
    data = np.loadtxt('processed_data/data.txt', dtype=float)
    user_Id_index = pickle.load(open('processed_data/user_id_index.pkl', 'rb'), encoding='bytes')
    movie_Id_index = pickle.load(open('processed_data/movie_id_index.pkl', 'rb'), encoding='bytes')

    ratio = 0.6
    num_rows = data.shape[0]
    train_data = data[:int(ratio*num_rows)]
    vali_data = data[int(ratio*num_rows):int((ratio+(1-ratio)/2)*num_rows)]
    test_data = data[int((ratio+(1-ratio)/2)*num_rows):]

    NUM_USERS = max(user_Id_index.values()) + 1
    NUM_MOVIES = max(movie_Id_index.values()) + 1


    R = np.zeros([NUM_USERS, NUM_MOVIES])
    for row in train_data:
        R[int(row[0]), int(row[1])] = float(row[2])

    
    pmf = PMF(R=R, iters=1000, lr=0.001, lambda_u=0.01, lambda_v=0.01, K=20)
        
    pmf.train(vali_data=vali_data)

    pred = pmf.predict(test_data)
    rmse = get_rmse(pred=pred, true=test_data[:,2])
    print('Test RMSE: ' + str(rmse))

    pmf.plt()
