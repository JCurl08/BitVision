import string
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QHBoxLayout, \
    QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
import pickle

import cv2
import numpy as np
import mediapipe as mp

from controller.Controller import Controller
model_path = "../models/pose_model.pkl"

class ImageProcessingThread(QThread):
    # Create a signal to send the processed images to the main thread
    update_image = pyqtSignal((np.ndarray, string))

    def run(self):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Instantiate controller
        try:
            with open(model_path, "rb") as f:
                controller = Controller(pickle.load(f))
            print("Successfully loaded model")
        except IOError:
            print("failed to open model")
            return

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
                if results != None and results.pose_landmarks != None:
                    row = []
                    for landmark in results.pose_landmarks.landmark:
                        row.append(landmark.x)
                        row.append(landmark.y)
                        row.append(landmark.z)
                    action = controller.do_action(row)

                # draw 3D pose landmarks live
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                self.update_image.emit(image, action)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.image_label = QLabel()
        self.controller_label = QLabel()
        self.start_stop_button = QPushButton("Start capture")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        left_layout = QVBoxLayout()  # For the first image and the button
        left_layout.addWidget(self.image_label)
        # left_layout.addWidget(self.controller_label)  # Add the second image below

        main_layout = QHBoxLayout()  # Main layout to hold everything
        main_layout.addLayout(left_layout)  # Add the top layout
        main_layout.addWidget(self.start_stop_button)


        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set up QTimer and connect the start/stop button
        self.start_stop_button.clicked.connect(self.toggle_stream)
        self.is_streaming = False  # To track the streaming state

        # Start the image processing thread
        self.thread = ImageProcessingThread()
        self.thread.update_image.connect(self.update_image)
        self.thread.start()

    def toggle_stream(self):
        if self.is_streaming:
            self.start_stop_button.setText("Start capture")
        else:
            self.start_stop_button.setText("Stop capture")
        self.is_streaming = not self.is_streaming

    @pyqtSlot((np.ndarray, string))
    def update_image(self, image):
        # Update the QLabel with the new image
        resized_image = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        height, width, channels = resized_image.shape
        bytes_per_line = channels * width
        flipped_image = cv2.flip(resized_image, 1)
        cvt_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        qimage = QImage(cvt_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(qpixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())