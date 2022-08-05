import os

import pandas as pd

from src.PoseCapturer import PoseEstimator
from src.STGCN import STGCN
from src.TrainData import TrainingData


class ExerciseTracker:
    def __init__(self,train=False):
        print('Initialized')
        if train:
            self.model=self.trainModel()
        else:
            self.model=STGCN()
            self.model.load('STGCN_Model.pth')

    def start(self):
        print('Started')

    def trainModel(self):
        training_data=TrainingData('C:\Project DBs\Final Research DB')
        self.model = STGCN()
        for epoch in range(0,100):
            self.model.train(x="C:\Project DBs\Final Research DB\\final_data.npy",y='C:\Project DBs\Final Research DB\\final_data_label.pkl')
        self.model.save("C:\Project DBs\Final Research DB\\STGCN_Model.pth")
        self.model.test(x="C:\Project DBs\Final Research DB\\test_data.npy",y='C:\Project DBs\Final Research DB\\test_data_label.pkl')

    def track(self):
        pose_estimator=PoseEstimator()
        pose_estimator.capture(self.model)


