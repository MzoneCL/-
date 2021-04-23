from lara import LARA
from my_dataset import MyDataSet
from torch.utils import data

if __name__ == '__main__':

    train_dataset = MyDataSet(data_path='data/train/train_data.csv', usr_emb_path='data/train/user_emb.csv')
    neg_dataset = MyDataSet(data_path='data/train/neg_data.csv', usr_emb_path='data/train/user_emb.csv')

    lara = LARA()

    lara.train(train_dataset=train_dataset, neg_dataset=neg_dataset)