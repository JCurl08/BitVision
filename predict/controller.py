import pyautogui


class Controller:

    def __init__(self, predictor, button):
        self.predictor = predictor
        self.button = button

    def do_press(self):
        pyautogui.press(self.button)