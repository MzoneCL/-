import numpy as np
import copy
from evaluation import get_rmse
import matplotlib.pyplot as plt

# 参考博客：https://zhuanlan.zhihu.com/p/34422451
class PMF():
    def __init__(self, R, lambda_u = 0.01, lambda_v = 0.01, lr = 0.001, K = 40, iters = 200):
        self.R = R # 评分矩阵
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lr = lr # 学习率
        self.K = K # 潜矩阵列数
        self.iters = iters # 迭代轮数

        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1

        # 都乘以 0.1 是为了防止值过大，造成 overflow 错误（乘之前就遇到了 RuntimeWarning: overflow encountered ）
        self.U = 0.1*np.random.random((self.R.shape[0], K))
        self.V = 0.1*np.random.random((self.R.shape[1], K))

        self.list_loss = []
        self.list_rmse = []
        self.list_iters = []

    def train(self, vali_data):

        last_rmse = None
        
        for iter in range(self.iters):
            grad_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_u*self.U
            grad_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_v*self.V

            self.U = self.U - self.lr * grad_u
            self.V = self.V - self.lr * grad_v
            
            rmse = get_rmse(self.predict(vali_data), vali_data[:,2])
            if last_rmse != None and last_rmse - rmse <= 0:
                print('Convergence at iter: ' + str(iter))
                break
            else:
                last_rmse = rmse

            if iter % 10 == 0:
                cur_loss = self.compute_loss()
                print('Iter ' + str(iter) + ' loss: ' + str(cur_loss) + '   RMSE: ' + str(rmse))
                self.list_loss.append(cur_loss)
                self.list_rmse.append(rmse)
                self.list_iters.append(iter)

    def predict(self, data):
        index_data = data[:,0:2].astype(int) # 取前两列（用户索引 电影索引）
        u_features = self.U.take(index_data.take(0, axis=1), axis=0) # a.take([1,2,3], axis=0) 选择a的第一二三行数据
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        pred = np.sum(u_features*v_features, 1)
        return pred

    def compute_loss(self):
        loss1 = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2)
        loss2 = self.lambda_u*np.sum(np.square(self.U))
        loss3 = self.lambda_v*np.sum(np.square(self.V))

        return loss1 + loss2 + loss3

    def plt(self):    
        plt.subplot(1,2,1)
        plt.title('Iter - Loss')
        plt.plot(self.list_iters, self.list_loss)
        plt.subplot(1,2,2)
        plt.title('Iter - RMSE')
        plt.plot(self.list_iters, self.list_rmse, 'r')
        plt.savefig('运行结果/iter_loss_rmse.png')
        plt.show()

