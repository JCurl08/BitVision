from train.posepredictor import PosePredictor
import pandas as pd

def main():
    # TODO: add data collection
    train_data = pd.read_csv("../test_data/data.csv")  # using sample data for now

    pp = PosePredictor()
    pp.fit(train_data)
    pp.save("test")


if __name__ == "__main__":
    main()