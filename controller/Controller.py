import numpy as np
import pyautogui

class Controller:

    def __init__(self, predictor, control_scheme):
        self.predictor = predictor
        self.control_scheme = control_scheme

    def do_action(self, data) -> None:
        """
        :param data: the single frame data to classify
        :return: None
        if data matches the action it's predictor is looking for, performs key press on the specified key
        """
        y_hat = self.predictor.predict(np.reshape(data, (1, -1)))
        if y_hat:
            print("Activate: " + y_hat[0])
            pyautogui.press(self.control_scheme[y_hat[0]])