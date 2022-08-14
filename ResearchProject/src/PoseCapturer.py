import time

import cv2
import mediapipe as mp
import os

import numpy as np
import torch


class PoseEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def capture(self, model):

        cap=cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

        self.model = model
        workout = 'Test'
        font = cv2.FONT_HERSHEY_SIMPLEX
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            nodes = []
            frame = 0
            rep_count=0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)


                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


                nodes.append(results.pose_landmarks)

                if frame == 2*fps:
                    node_values=self.determine_node(nodes)
                    if node_values is not None:
                        val = self.model.predict(node_values)
                        if workout != val:
                            workout_time=rep_count*2
                            if workout!="Test":
                                print(f'{workout} done for {workout_time} seconds')
                            workout = val
                            rep_count=1
                        elif workout==val and workout!="Test":
                            rep_count=rep_count+1
                    frame = 0
                    nodes.clear()
                if workout != "Test":
                    cv2.putText(image,f'{workout} || Time: {rep_count*2}s',(50, 50),
                font, 1,
                (0, 255, 255),
                1,
                cv2.LINE_4)


                cv2.imshow('MediaPipe Pose', image)
                frame = frame + 1
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def break_out(self, values):
        for landmark in values.pose_landmarks.landmark:
            if landmark.visibility < 0.5:
                return True

    def capture_from_training_data(self, file_path):
        return_val = []
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            return_val.append(results.pose_landmarks)
        pose.close()
        cap.release()
        return self.determine(return_val)

    def determine_node(self, results):
        try:
            return_val = []
            for result in results:
                frames = []
                for V in range(0, 33):
                    x = [result.landmark[V].x]
                    y = [result.landmark[V].y]
                    z = [result.landmark[V].z]
                    frames.append([x,y,z])
                return_val.append(frames)
            return torch.tensor([return_val])
        except AttributeError:
            print('Not all limbs are visible in the camera. Kindly readjust yourself and try again.')
            return None
        except Exception:
            return None
    def determine(self, result):
        try:
            results = []
            for T in range(0, 200):
                frames = []
                for V in range(0, 33):
                    nodes = []
                    x = []
                    y = []
                    z = []
                    x.append(result[T].landmark[V].x)
                    y.append(result[T].landmark[V].y)
                    z.append(result[T].landmark[V].z)
                    nodes.append(x)
                    nodes.append(y)
                    nodes.append(z)
                    frames.append(nodes)
                results.append(frames)
            return np.array(results)
        except AttributeError:
            return None
        except IndexError:
            return None
        except Exception:
            return None
