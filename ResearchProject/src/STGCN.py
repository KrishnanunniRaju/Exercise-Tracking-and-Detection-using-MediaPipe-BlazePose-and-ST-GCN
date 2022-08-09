import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from src.Helpers.Feeder import Feeder
from src.Model import Model


class STGCN:
    def __init__(self,optimizer,labels):
        self.accuracy = 0
        self.loss = nn.CrossEntropyLoss()
        self.model = Model(in_channels=3, num_class=10, edge_importance_weighting=True)
        self.model.apply(weights_init)
        self.load_optimizer(optimizer, 0.01)
        self.dev = 'cpu'
        self.init_environment()
        self.label = labels


    def init_environment(self):
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def test(self, x, y, evaluation=True):
        y_pred = []
        y_true = []
        label = []
        self.model.eval()
        with open(y, 'rb') as f:
            labels = pickle.load(f)
            label = torch.tensor(labels).long().to(self.dev)

        data = torch.tensor(np.load(x)).float().to(self.dev)
        # inference
        with torch.no_grad():
            output = self.model(data)

        result, correct = self.determine(output, label)
        # get loss
        if evaluation:
            y_pred.append(result)
            y_true.append(label)
        print(f'Number of correct detections:{correct}. Ratio: {correct/len(labels)}')
        cf_matrix = confusion_matrix(label.detach().cpu().numpy(), np.array(result))
        print(cf_matrix)

    def train(self, x, y):
        self.model.train()
        dataset = Feeder(data_path=x, label_path=y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            drop_last=True)

        for epoch in range(0, 100):
            correct = 0
            for data, label in loader:
                # forward

                data = data.float().to(self.dev)
                label = label.long().to(self.dev)
                output = self.model.forward(data)

                loss = self.loss(output, label)
                _, value = self.determine(output, label)
                correct += value
                self.test(x="C:\Project DBs\Final Research DB\\test_data.npy",
                                y='C:\Project DBs\Final Research DB\\test_data_label.pkl')
                self.optimizer.zero_grad()
                # backward
                loss.backward()
                self.optimizer.step()

                # statistics
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                self.accuracy = 100 * correct / len(data)
                self.show_iter_info()
                print(f'Accuracy: {self.accuracy};[{correct}]')
                self.meta_info['iter'] += 1

    def determine(self, output, label):
        result = []
        correct = 0
        for item in output:
            #value = (item == torch.max(item)).nonzero().flatten().tolist()[0]
            value=(item == torch.max(item)).nonzero(as_tuple=True)[0].item()
            result.append(value)
        for idx in range(len(result)):
            if result[idx] == label[idx]:
                correct += 1
        return result, correct

    def predict(self, data):
        out = self.model.forward(data)
        result = (out == torch.max(out)).nonzero(as_tuple=True)[1].item()
        return self.label[result]

    def save(self, path,labelpath):
        torch.save(self.model.state_dict(), path)
        with open(labelpath, 'wb') as f:
            pickle.dump(self.label, f)

    def load(self, path,labels_path):
        self.model.load_state_dict(torch.load(path))
        with open(labels_path, 'rb') as f:
            self.label = pickle.load(f)
        self.model.eval()

    def show_iter_info(self):
        info = '\tIter {} Done.'.format(self.meta_info['iter'])
        for k, v in self.iter_info.items():
            if isinstance(v, float):
                info = info + ' | {}: {:.4f}'.format(k, v)
            else:
                info = info + ' | {}: {}'.format(k, v)

        print(info)

    def load_optimizer(self, name, learning_rate):
        if name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate)
        elif name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                nesterov=True)
        else:
            raise ValueError()
        self.lr = learning_rate


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)