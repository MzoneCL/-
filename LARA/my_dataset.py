import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MyDataSet(Dataset):
    # data_path - 训练数据集对应的路径（train_data.csv \ neg_data.csv）
    # usr_emb_path - 用户嵌入矩阵的路径（user_emb.csv）
    def __init__(self, data_path, usr_emb_path) -> None:
    
        self.data = pd.read_csv(data_path, header=None)
        self.usr_emb = pd.read_csv(usr_emb_path, header=None)

        self.usr = self.data.loc[:, 0]
        self.item = self.data.loc[:, 1]
        self.attr = self.data.loc[:, 2]
        self.usr_emb_values = np.array(self.usr_emb[:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        usr = self.usr[index]
        item = self.item[index]
        usr_emb = self.usr_emb_values[usr]

        attr = self.attr[index][1: -1].split() # 剔除 [ ]
        attr = torch.tensor(list([int(i) for i in attr]), dtype=torch.long) # 字符串转 int，再dtype = torch.long
        attr = np.array(attr)

        return usr, item, attr, usr_emb