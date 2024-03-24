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
    update_image = pyqtSignal(np.ndarray)
    update_controller_image = pyqtSignal(str)

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

                self.update_image.emit(image)
                self.update_controller_image.emit(str(action[0]))

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()



controller_images = {
    "neutral": "neutral.png",
    "crouch": "down.png",
    "walk_left": "left.png",
    "walk_right": "right.png",
    "jump_front": "jump.png",
    "jump_left": "jump_left.png",
    "jump_right": "jump_right.png",
    "pause": "start.png"
}
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #Preprocess images
        self.controller_pixmaps = {}
        self.preprocessImages()

        # Set up the UI
        self.image_label = QLabel()
        self.controller_label = QLabel()

        self.controller_label.setPixmap(self.controller_pixmaps["neutral"])
        self.start_stop_button = QPushButton("Start capture")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        bottom_layout = QHBoxLayout()  # For the first image and the button
        bottom_layout.addWidget(self.controller_label)  # Add the second image below
        bottom_layout.addWidget(self.start_stop_button)

        main_layout = QVBoxLayout()  # Main layout to hold everything
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(bottom_layout)  # Add the top layout

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set up QTimer and connect the start/stop button
        self.start_stop_button.clicked.connect(self.toggle_stream)
        self.is_streaming = False  # To track the streaming state

        # Start the image processing thread
        self.thread = ImageProcessingThread()
        self.thread.update_image.connect(self.update_image)
        self.thread.update_controller_image.connect(self.update_controller_image)
        self.thread.start()

    def preprocessImages(self):
        for action, filename in controller_images.items():
            print(filename)
            img = cv2.imread("./images/" + filename)
            self.controller_pixmaps[action] = self.cv2_to_qpixmap(img)


    def toggle_stream(self):
        if self.is_streaming:
            self.start_stop_button.setText("Start capture")
        else:
            self.start_stop_button.setText("Stop capture")
        self.is_streaming = not self.is_streaming

    @pyqtSlot(np.ndarray)
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

    @pyqtSlot(str)
    def update_controller_image(self, action):
        self.controller_label.setPixmap(self.controller_pixmaps[action])

    def cv2_to_qpixmap(self, cv_image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

        # Convert numpy array to QImage
        height, width, channel = resized_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        qpixmap = QPixmap.fromImage(q_image)

        return qpixmap

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())