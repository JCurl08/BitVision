from train.posepredictor import PosePredictor
import pandas as pd
import glob

def preprocess_training_data(data, label):
    data["class"] = label
    return data

def preprocess_all():
    prefix = "training_data"
    suffix = ".csv"
    path = prefix + "/*" + suffix

    frames = []
    for fname in glob.glob(path):
        label = fname[len(prefix) + 1:-len(suffix)]
        if (label == "training_data_all"):
            continue
        data = preprocess_training_data(pd.read_csv("./" + fname, header=None), label)
        frames.append(data)

    training_data_all = pd.concat(frames)
    training_data_all.to_csv("./training_data/training_data_all.csv")

def main():
    preprocess_all()

    train_data = pd.read_csv("./training_data/training_data_all.csv")
    pp = PosePredictor()
    pp.fit(train_data)
    pp.save("pose_model")

if __name__ == "__main__":
    main()