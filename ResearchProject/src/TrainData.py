import cv2
import pandas as pd
import os
from src.PoseCapturer import PoseEstimator
class TrainingData:
    def __init__(self, folder_path):
        self.df = None
        self.db_folder = folder_path
        self.files = []
        self.labels = []
        self.CONST_DATAFOLDER_PATH = 'data'
        self.generate_frames()


    def read_video_files_into_dataframe(self, path):
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        label = os.path.abspath(parent_path).split("_")[2]
        for file in os.listdir(path):
            if file.endswith(".mp4"):
                self.files.append(os.path.join(path, file))
                self.labels.append(label)

    def read_files(self):
        for iteration, file in enumerate(os.listdir(self.db_folder)):
            d = os.path.join(self.db_folder, file)
            if os.path.isdir(d):
                self.read_video_files_into_dataframe(os.path.join(d, self.CONST_DATAFOLDER_PATH))
        self.df = pd.DataFrame(list(zip(self.files, self.labels)),
                               columns=['Name', 'Label'])

    def generate_frames(self):
        self.read_files()
        skeleton_values=[]
        pose_estimator=PoseEstimator()
        for idx,row in self.df.iterrows():
            print(f'Generating training data for {row["Name"]}')
            result=pose_estimator.capture_from_training_data(row['Name'])
            print(result)
            skeleton_values.append(result)
            print(result)
        self.df['Data']=skeleton_values
        final_file=os.path.join(self.db_folder,'final_data.csv')
        self.df.to_csv(final_file)



