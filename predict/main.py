import sys
import time

import cv2
import numpy as np
from predict.Predictor import Predictor
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = '../models/pose_landmarker_heavy.task'

model_file_names = [
    "test.pkl"
]

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def main():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the live stream mode:
    def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    #     cv2.imshow('Show', annotated_image)
    #     cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print('pose landmarker result: {}'.format(result))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    vid = cv2.VideoCapture(1)
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = vid.read()

            # the 'q' button is set as the quitting button you may use any desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # The results are accessible via the `result_callback` provided in the `PoseLandmarkerOptions` object.
            # The pose landmarker must be created with the live stream mode.
            frame_timestamp_ms = int(time.time()*1000.0)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            # Display the resulting frame
            cv2.imshow('frame', frame)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
