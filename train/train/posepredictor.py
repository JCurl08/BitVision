from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd
import pickle


class PosePredictor:

    def __init__(self):
        self.rf = RandomForestClassifier()

    def fit(self, X, y):
        """ Function to fit the data from collect"""
        train_data = pd.read_csv("../test_data/data.csv")  # using sample data for now
        X = train_data.drop("class", axis=1)
        y = train_data["class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.rf.fit(X_train, y_train)

        y_pred = self.rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def save(self, model_name):
        path = "../models/" + model_name + ".pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(self.rf, f)
        except IOError:
            print("failed to save model")
        print("saved model \"" + model_name + "\" successfully")
