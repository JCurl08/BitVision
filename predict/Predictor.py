from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

"""
Returns one prediction.

TODO: Update predict to return a tuple/dict/array of predictions for 1 frame

input_models = list of model file names
"""


class Predictor:
    """

    input_models = list of model names

    """

    def __init__(self, input_models):
        self.sklearn_models = []

        for model in input_models:
            try:
                with open("../models/" + model, "rb") as f:
                    self.sklearn_models.append(pickle.load(f))
                print("Successfully loaded model:" + model)
            except IOError:
                print("failed to open model:" + model)

    """
    Returns one prediction.

    TODO: Update predict to return a tuple/dict/array of predictions for 1 frame
    """

    def predict(self, input_sample):
        # Go through models
        for model in self.sklearn_models:
            input_array = np.array(input_sample)
            prediction = model.predict(input_array)
            return prediction
