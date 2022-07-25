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
        training_data=TrainingData('C:\Project DBs\Final Research DB')
        print(training_data.db_folder)
        posecapture = PoseEstimator()
        folder_path = 'C:\\Project DBs\\Final Research DB\\FinalDB'
        posecapture.capture_from_training_data(folder_path)

