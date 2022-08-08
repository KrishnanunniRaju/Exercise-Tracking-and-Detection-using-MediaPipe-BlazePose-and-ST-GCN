import os

import pandas as pd

from src.PoseCapturer import PoseEstimator
from src.STGCN import STGCN
from src.TrainData import TrainingData


class ExerciseTracker:
    def __init__(self, train=False):
        print('Initialized')
        if train:
            self.start()
            self.model = self.trainModel(generate=True)
        else:
            self.model = STGCN('Adam')
            self.model.load("C:\Project DBs\Final Research DB\\STGCN_Model_SGD.pth","C:\Project DBs\Final Research DB\\STGCN_Model_Label.pkl")

    def start(self):
        print('Initializing Exercise tracker...\n Generating data and training the model.')

    def trainModel(self, generate=True):
        if generate:
            training_data=TrainingData('C:\Project DBs\Final Research DB')
            self.model = STGCN('Adam',training_data.labels)
            self.model.train(x="C:\Project DBs\Final Research DB\\final_data.npy",y='C:\Project DBs\Final Research DB\\final_data_label.pkl')
            self.model.save("C:\Project DBs\Final Research DB\\STGCN_Model_SGD.pth","C:\Project DBs\Final Research DB\\STGCN_Model_Label.pkl")
            self.model.test(x="C:\Project DBs\Final Research DB\\test_data.npy", y='C:\Project DBs\Final Research DB\\test_data_label.pkl')

    def test_model(self):
        self.model.test(x="C:\Project DBs\Final Research DB\\test_data.npy",
                        y='C:\Project DBs\Final Research DB\\test_data_label.pkl')

    def track(self):
        pose_estimator = PoseEstimator()
        pose_estimator.capture(self.model)
