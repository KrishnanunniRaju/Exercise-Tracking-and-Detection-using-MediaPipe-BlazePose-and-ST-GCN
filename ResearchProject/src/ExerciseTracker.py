import os
import pickle

import pandas as pd

from src.PoseCapturer import PoseEstimator
from src.STGCN import STGCN
from src.TrainData import TrainingData


class ExerciseTracker:
    def __init__(self, train=False,strategy='spatial',edge_importance=True):
        print('Initialized')
        if train:
            self.start()
            self.model = self.trainModel()
        else:
            with open(f'C:\Project DBs\Final Research DB\\Final_Model_{strategy}_{edge_importance}_Label.pkl','rb') as f:
                labels=pickle.load(f)
                self.model = STGCN('Adam',list(labels),strategy=strategy)
            self.model.load(f'C:\Project DBs\Final Research DB\\Final_Model_{strategy}_{edge_importance}.pth',f'C:\Project DBs\Final Research DB\\Final_Model_{strategy}_{edge_importance}_Label.pkl')

    def start(self):
        print('Initializing Exercise tracker...\nGenerating data and training the model.')

    def trainModel(self, generate=True):
        if generate:
            strategy=['uniform','spatial']
            edge_importance=[True,False]
            training_data=TrainingData('C:\Project DBs\Final Research DB')
            for strat in strategy:
                for ei in edge_importance:
                    self.model = STGCN('Adam',training_data.labels,strategy=strat,edge_importance=ei)
                    self.model.train(x="C:\Project DBs\Final Research DB\\final_data.npy",y='C:\Project DBs\Final Research DB\\final_data_label.pkl')
                    self.model.save(f'C:\Project DBs\Final Research DB\\Final_Model_{strat}_{ei}.pth',f'C:\Project DBs\Final Research DB\\Final_Model_{strat}_{ei}_Label.pkl')
                    self.test_model()

    def test_model(self):
        self.model.test(x="C:\Project DBs\Final Research DB\\test_data.npy",
                        y='C:\Project DBs\Final Research DB\\test_data_label.pkl')

    def track(self):
        pose_estimator = PoseEstimator()
        pose_estimator.capture(self.model)
