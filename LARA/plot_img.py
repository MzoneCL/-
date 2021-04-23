from evaluation import p_at_k
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 画 p@10  p@20   ndcg@10   ndcg@20 图

def plt_img():
    res = pd.read_csv('data/result/test_result.csv')
    p_at_10 = res['p@10']
    p_at_20 = res['p@20']
    ndcg_at_10 = res['ndcg@10']
    ndcg_at_20 = res['ndcg@20']

    x = np.linspace(1, 400, 400)

    plt_p10(x, p_at_10)
    plt_p20(x, p_at_20)
    plt_n10(x, ndcg_at_10)
    plt_n20(x, ndcg_at_20)
    

def plt_p10(x, p_at_10):
    plt.title('p@10')
    plt.plot(x, p_at_10)
    plt.savefig('img/p_at_10.jpg')
    plt.cla()

def plt_p20(x, p_at_20):
    plt.title('p@20')
    plt.plot(x, p_at_20)
    plt.savefig('img/p_at_20.jpg')
    plt.cla()

def plt_n10(x, ndcg_at_10):
    plt.title('NDCG@10')
    plt.plot(x, ndcg_at_10)
    plt.savefig('img/ndcg_at_10.jpg')
    plt.cla()

def plt_n20(x, ndcg_at_20):
    plt.title('NDCG@10')
    plt.plot(x, ndcg_at_20)
    plt.savefig('img/ndcg_at_20.jpg')
    plt.cla()

if __name__ == '__main__':
    plt_img()
