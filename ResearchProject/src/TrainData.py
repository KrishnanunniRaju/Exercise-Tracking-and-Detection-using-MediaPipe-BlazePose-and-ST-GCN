import pickle
import random

import cv2
import numpy as np
import pandas as pd
import os
from src.PoseCapturer import PoseEstimator


class TrainingData:
    def __init__(self, folder_path):
        self.df = None
        self.db_folder = folder_path
        self.files = []
        self.labels = []
        self.final_data_file = "C:\Project DBs\Final Research DB\\final_data.npy"
        self.final_label_path = 'C:\Project DBs\Final Research DB\\final_data_label.pkl'
        self.test_data_file = "C:\Project DBs\Final Research DB\\test_data.npy"
        self.test_label_path = 'C:\Project DBs\Final Research DB\\test_data_label.pkl'
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
        skeleton_values = []
        pose_estimator = PoseEstimator()
        final_labels = []
        labels = self.df['Label'].unique().tolist()
        for idx, row in self.df.iterrows():
            print(f'Generating training data for {row["Name"]}')
            result = pose_estimator.capture_from_training_data(row['Name'])
            if result is not None:
                skeleton_values.append(result)
                final_labels.append(labels.index(row['Label']))
        final_data = [skeleton_values, final_labels]
        random.shuffle(final_data)
        training_data_values = np.array(final_data[0][0:899])
        training_data_labels = final_data[1][0:899]
        testing_data_values = np.array(final_data[0][900:len(final_data[0])])
        testing_data_label = final_data[1][900:len(final_data[1])]

        np.save(self.final_data_file, training_data_values)
        np.save(self.test_data_file, testing_data_values)

        with open(self.final_label_path, 'wb') as f:
            pickle.dump(training_data_labels, f)

        with open(self.test_label_path, 'wb') as f:
            pickle.dump(testing_data_label, f)

        self.labels = labels
