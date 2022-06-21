import cv2
import pandas as pd
import os


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
        final_folder = f'{self.db_folder}\\FinalDB'
        for index in range(0, len(self.df)):
            vidcap = cv2.VideoCapture(self.df.iloc[index][0])
            success, image = vidcap.read()
            count = 0
            os.makedirs(f'{final_folder}\\{self.df.iloc[index][1]}\\{index}')
            while success:
                file_name = f'{final_folder}\\{self.df.iloc[index][1]}\\{index}\\frame_{count}.jpg'
                print(file_name)
                cv2.imwrite(file_name, image)  # save frame as JPEG file.
                success, image = vidcap.read()
                count = count + 1
                print('Read a new frame: ', success)
        print(f'The final project database can be found in {final_folder}')


if __name__ == '__main__':
    instance = TrainingData('C:\Project DBs\Final Research DB')
