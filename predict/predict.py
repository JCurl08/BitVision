from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

"""
Returns one prediction.

TODO: Update predict to return a tuple/dict/array of predictions for 1 frame

input_models = list of model file names
"""


def predict(self, input_sample, input_models):
    # Load models from string names
    sklearn_models = []
    for model in input_models:
        try:
            with open(model, "rb") as f:
                sklearn_models.append(pickle.load(f))
        except IOError:
            print("failed to open model:" + model)

    # Go through models
    for model in sklearn_models:
        input_array = np.array(input_sample)
        prediction = model.predict(input_array)
        return prediction
