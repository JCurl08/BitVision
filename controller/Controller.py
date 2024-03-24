import numpy as np
import pyautogui

class Controller:

    def __init__(self, predictor, control_scheme):
        self.predictor = predictor
        self.control_scheme = control_scheme
        self.current_action = 'neutral'

    def do_action(self, data):
        """
        :param data: the single frame data to classify
        :return: None
        if data matches the action it's predictor is looking for, performs key press on the specified key
        """
        y_hat = self.predictor.predict(np.reshape(data, (1, -1)))[0]
        if y_hat and y_hat != 'neutral':
            print("Activate: " + y_hat)

            if y_hat == self.current_action:
                # print("Press and hold: " + " ".join(k) for k in self.control_scheme[y_hat])
                for key in self.control_scheme[y_hat]:
                    pyautogui.keyDown(key)
            else:
                if self.current_action:
                    # print("Release: " + " ".join(k) for k in self.control_scheme[y_hat])
                    for key in self.control_scheme[self.current_action]:
                        pyautogui.keyUp(key)
                # print("Press: " + " ".join(k) for k in self.control_scheme[y_hat])
                for key in self.control_scheme[y_hat]:
                    pyautogui.keyDown(key)
            self.current_action = y_hat
        else:
            for key in self.control_scheme[self.current_action]:
                pyautogui.keyUp(key)
            self.current_action = 'neutral'