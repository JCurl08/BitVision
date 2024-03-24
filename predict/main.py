import sys
import time

import cv2
import numpy as np
from predict.Predictor import Predictor
import mediapipe as mp

model_path = '../models/pose_landmarker_heavy.task'

model_file_names = [
    "test.pkl"
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
def main():
    vid = cv2.VideoCapture(1)
    with mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
    ) as pose:
        while vid.isOpened():
            # read webcam image
            success, image = vid.read()

            # skip empty frames
            if not success:
                continue

            # calculate pose
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # draw 3D pose landmarks live
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            print('pose landmarker result: {}'.format(results.pose_landmarks))

            # draw image
            cv2.imshow("MediaPipePose", cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
