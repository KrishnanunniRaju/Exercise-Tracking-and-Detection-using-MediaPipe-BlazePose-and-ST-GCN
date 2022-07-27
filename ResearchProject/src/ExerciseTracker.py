import os

import pandas as pd

from src.PoseCapturer import PoseEstimator
from src.TrainData import TrainingData


class ExerciseTracker:
    def __init__(self,train=False):
        print('Initialized')
        if train:
            self.model=self.trainModel()

    def start(self):
        print('Started')

    def trainModel(self):
        db_folder='C:\Project DBs\Final Research DB'
        #training_data=TrainingData('C:\Project DBs\Final Research DB')
        #print(training_data.db_folder)
        file=os.path.join(db_folder,"final_data.csv")
        df=pd.read_csv(file)
        print(len(df))


