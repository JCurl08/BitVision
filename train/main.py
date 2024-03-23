from train.posepredictor import PosePredictor

def main():
    pp = PosePredictor()
    pp.fit(None, None)
    pp.save("test")


if __name__ == "__main__":
    main()