from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd


class Train:
    def __init__(self):
        self.collect()
        self.fit()

    # Function collects data from open pose
    def collect(self):
        """ Collects image and pose data """
        pass

    def fit(self):
        """ Function to fit the data from collect"""
        train_data = pd.read_csv("../test_data/data.csv")  # using sample data for now
        X = train_data.drop("class", axis=1)
        y = train_data["class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        pass
