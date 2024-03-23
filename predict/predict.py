from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def predict(self, input_sample, input_models):
    for model in input_models:
        with open(model, 'wb') as f:
            pickle.loads()

    clf = pickle.loads(input_model)
    input_array = np.array(input_sample)
    prediction = clf.predict(input_array)
    return prediction
