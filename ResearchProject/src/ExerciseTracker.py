import Model
import PoseCapturer
import os
import TrainData

class ExerciseTracker:
    def __init__(self):
        print('Initialized')
        self.model=self.trainModel()

    def start(self):
        print('Started')

    def trainModel(self):
        #training_data=TrainData()
        #print(training_data.db_folder)
        return 'model'

exercisetracker=ExerciseTracker()
exercisetracker.trainModel()