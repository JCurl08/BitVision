import numpy as np
import pyautogui
feature_head = np.array([i for i in range(100)])

class Controller:

    def __init__(self, predictor):
        self.predictor = predictor

    def do_action(self, data) -> None:
        """
        :param data:
        :return: None
        if data matches the action it's predictor is looking for, performs key press on this controller's key
        """
        y_hat = self.predictor.predict(np.append(feature_head, data, axis=0))
        if y_hat:
            print("Activate: " + y_hat)
            # pyautogui.press(self.button)