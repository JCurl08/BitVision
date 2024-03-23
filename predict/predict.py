from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def predict(self, input_sample, input_file):
    clf = pickle.loads(input_file)
    input_array = np.array(input_sample)
    prediction = clf.predict(input_array)
    return prediction
