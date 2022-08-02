import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.Helpers.Feeder import Feeder
from src.Model import Model


class STGCN:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()
        self.model=Model(in_channels=3,num_class=10,edge_importance_weighting=False)
        self.load_optimizer('Adam',0.1)
        self.dev='cpu'
        self.init_environment()

    def init_environment(self):
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def train(self,x,y):
        self.data_loader = dict()
        dataset = Feeder(data_path=x, label_path=y)
        loader=self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset,
                batch_size=11,
                shuffle=True,
                drop_last=True)
        for data,label in loader:
            #forward
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            output = self.model.forward(data)
            loss = self.loss(output,label)
            self.optimizer.zero_grad()
            #backward
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1

    def predict(self,x):
        out=self.model.forward(x)
        print((out == torch.max(out)).nonzero(as_tuple=True)[1].item())


    def save(self,path):
        torch.save(self.model.state_dict(),path)

    def load(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def show_iter_info(self):
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            print(info)


    def load_optimizer(self,name,learning_rate):
        if name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate)
            self.lr=learning_rate
        else:
            raise ValueError()
