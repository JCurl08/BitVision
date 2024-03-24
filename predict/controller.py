import pyautogui


class Controller:

    def __init__(self, predictor, button):
        self.predictor = predictor
        self.button = button

    def do_action(self, data):
        """
        :param data:
        :return: None
        if data matches the action it's predictor is looking for, performs key press on this controller's key
        """
        if self.predictor.predict(data):
            pyautogui.press(self.button)