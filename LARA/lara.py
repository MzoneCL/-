import time
from torch.utils import data
import torch
from torch import nn, optim
import evaluation

def init_param(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.xavier_normal_(module.bias.unsqueeze(0))
    else:
        pass    


class Generator(nn.Module):
    def __init__(self, attr_num, attr_present_dim, hidden_dim, user_emb_dim):
        super(Generator, self).__init__()

        self.attr_num = attr_num
        self.attr_present_dim = attr_present_dim
        self.user_emb_dim = user_emb_dim

        self.embedding = nn.Embedding(2*attr_num, attr_present_dim) # 嵌入层，将每一个属性映射成一个 attr_present_dim 维的向量
        self.l1 = nn.Linear(attr_num*attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.active = nn.Tanh()

        self.__init_param__()

    def __init_param__(self):
        for module in self.embedding.modules():
            nn.init.xavier_normal_(module.weight)
        for module in self.modules():
            init_param(module)

    def forward(self, attr):
        attr_present = self.embedding(attr)
        attr_feature = torch.reshape(attr_present, [-1, self.attr_num*self.attr_present_dim])  # 将 embedding 后的每一个 item 的特征压到同一行

        out1 = self.active(self.l1(attr_feature))
        out2 = self.active(self.l2(out1))
        out3 = self.active(self.l3(out2))

        return out3    


class Discriminator(nn.Module):
    def __init__(self, attr_num, attr_present_dim, hidden_dim, user_emb_dim):
        super(Discriminator, self).__init__()

        self.attr_num = attr_num
        self.hidden_dim = hidden_dim
        self.user_emb_dim = user_emb_dim
        self.attr_present_dim = attr_present_dim

        self.embedding = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim + user_emb_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.active = nn.Tanh()

        self.__init_param__()

    def __init_param__(self):
        for module in self.embedding.modules():
            nn.init.xavier_normal_(module.weight)
        for module in self.modules():
            init_param(module)

    def forward(self, attr, user_emb):
        attr = attr.long()
        attr_present = self.embedding(attr)
        attr_feature = torch.reshape(attr_present, [-1, self.attr_num*self.attr_present_dim])  # 将 embedding 后的每一个 item 的特征压到同一行 
        emb = torch.cat((attr_feature, user_emb), 1) # 将用户向量和物品的特征向量进行拼接
        emb = emb.float()

        out1 = self.active(self.l1(emb))
        out2 = self.active(self.l2(out1))
        out3 = self.l3(out2)

        prob = torch.sigmoid(out3)

        return prob, out3 # 返回概率值


class LARA:
    def __init__(self, alpha=0, attr_num=18, attr_present_dim=5, batch_size=1024, hidden_dim=100, 
    user_emd_dim=18, learning_rate=0.0001, epoch=400):
        self.g = Generator(attr_num=attr_num, attr_present_dim=attr_present_dim,hidden_dim=hidden_dim,user_emb_dim=user_emd_dim)
        self.d = Discriminator(attr_num=attr_num, attr_present_dim=attr_present_dim,hidden_dim=hidden_dim,user_emb_dim=user_emd_dim)
        self.alpha = alpha # 正则项参数
        self.attr_num = attr_num # 属性数
        self.attr_present_dim = attr_present_dim # 物品属性维度（每一个属性扩张成的维度）
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim 
        self.user_emd_dim = user_emd_dim
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_dataset, neg_dataset):
        self.g = self.g.to(self.device)
        self.d = self.d.to(self.device)
        print("Train on ", self.device)

        loss = nn.BCELoss() # 用于二分类的交叉熵损失函数

        # 定义 生成器 和 判别器 的 优化器
        g_optim = optim.Adam(self.g.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        d_optim = optim.Adam(self.d.parameters(), lr=self.learning_rate, weight_decay=self.alpha)

        start = time.time()

        # 定义训练数据加载器
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=0)
        neg_loader = data.DataLoader(neg_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        dataLen = len(neg_dataset)

        list_d_loss = []
        list_g_loss = []

        for cur_epoch in range(self.epoch):
            i = 0
            # 获取 neg_loader 迭代器
            neg_iter = neg_loader.__iter__()
            
            # 训练判别器
           
            for train_user, train_item, train_attr, train_user_emb in train_loader:
                if i*self.batch_size >= dataLen:
                    break

                # 取出负采样的样本
                neg_user, neg_item, neg_attr, neg_user_emb = neg_iter.next()
                neg_attr = neg_attr.to(self.device)
                neg_user_emb = neg_user_emb.to(self.device)

                train_attr = train_attr.to(self.device)
                train_user_emb = train_user_emb.to(self.device)

                generated_user_emb = self.g(train_attr) # 使用生成器生成的用户表达

                prob_real, out3_real = self.d(train_attr, train_user_emb)
                prob_generated, out3_generated = self.d(train_attr, generated_user_emb)
                prob_neg, out3_neg = self.d(neg_attr, neg_user_emb)

                # 判别器的 loss 由三部分构成
                d_loss_real = loss(prob_real, torch.ones_like(prob_real))
                d_loss_generated = loss(prob_generated, torch.zeros_like(prob_generated))
                d_loss_neg = loss(prob_neg, torch.zeros_like(prob_neg))
                
                d_loss = torch.mean(d_loss_real + d_loss_generated + d_loss_neg)
                list_d_loss.append(d_loss)

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                generated_user_emb = self.g(train_attr) # 使用生成器生成的用户表达
                prob_generated, out3_generated = self.d(train_attr, generated_user_emb)
                g_loss = loss(prob_generated, torch.ones_like(prob_generated))
                list_g_loss.append(g_loss)

                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()

                i += 1

            print('Epoch:%d     耗时：%.2fs     d_loss: %.5f     g_loss: %.5f ' % (cur_epoch+1, (time.time() - start), d_loss, g_loss))

            start = time.time()  

            # 测试
            item, attr = evaluation.get_test_data()
            item = item.to(self.device)
            attr = attr.to(self.device)
            item_user = self.g(attr) # 使用生成器生成用户表示
            evaluation.to_valuate(item=item, item_user=item_user)
            g_optim.zero_grad()


            # 保存模型
            if cur_epoch % 10 == 0:
                torch.save(self.g.state_dict(), 'data/result/g_' + str(cur_epoch) + '.pt')
                torch.save(self.d.state_dict(), 'data/result/d_' + str(cur_epoch) + '.pt')