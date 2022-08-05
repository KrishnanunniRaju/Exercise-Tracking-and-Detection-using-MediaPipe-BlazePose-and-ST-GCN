import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from src.Helpers.Feeder import Feeder
from src.Model import Model


class STGCN:
    def __init__(self):
        self.accuracy = 0
        self.loss = nn.CrossEntropyLoss()
        self.model=Model(in_channels=3,num_class=10,edge_importance_weighting=False)
        self.load_optimizer('Adam',0.1)
        self.dev='cpu'
        self.init_environment()
        self.label = ['armraise', 'bicyclecrunch', 'birddog', 'curl', 'fly', 'legraise', 'overheadpress', 'pushup', 'squat',
                 'superman']


    def init_environment(self):
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def test(self,x,y, evaluation=True):
        y_pred = []
        y_true = []
        self.model.eval()
        dataset = Feeder(data_path=x, label_path=y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            drop_last=True)
        loss_value = []
        result_frag = []
        label_frag = []
        correct=0
        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                correct += (output == label).float().sum()

                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
                y_pred.append(output)
                y_true.append(label)



        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)

    def train(self,x,y):
        self.model.train()
        dataset = Feeder(data_path=x, label_path=y)
        loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=256,
                shuffle=True,
                drop_last=True)
        correct=0
        for data,label in loader:
            #forward
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            output = self.model.forward(data)
            loss = self.loss(output,label)
            correct += (output == label).float().sum()
            self.optimizer.zero_grad()
            #backward
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.accuracy = 100 * correct / len(data)
            self.show_iter_info()
            print(f'Accuracy: {self.accuracy}')
            self.meta_info['iter'] += 1

    def predict(self,x):
        out=self.model.forward(x)
        result=(out == torch.max(out)).nonzero(as_tuple=True)[1].item()
        return self.label[result]

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
