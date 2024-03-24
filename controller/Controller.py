import numpy as np
import pyautogui

class Controller:

    def __init__(self, predictor):
        self.predictor = predictor

    def do_action(self, data):
        """
        :param data:
        :return: None
        if data matches the action it's predictor is looking for, performs key press on this controller's key
        """
        y_hat = self.predictor.predict(np.reshape(data, (1, -1)))
        if y_hat:
            print("Activate: " + y_hat)
            return y_hat
            # pyautogui.press('w')