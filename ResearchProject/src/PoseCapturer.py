import cv2
import mediapipe as mp
import os

import numpy as np


class PoseEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def capture(self):
        cap = cv2.VideoCapture(0)
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def capture_from_training_data(self, file_path):
        return_val=[]
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

    def determine(self,result):
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

