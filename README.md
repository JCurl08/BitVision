# BitVision

BitVision is a Python-based computer vision app that allows users to record actions (3D poses) on video using Mediapipe and map them to keyboard inputs. Training data is associated with actions using random forest classification. 
During live recording, recorded video frames are processed against the random forest model, generating a set of key presses to perform according to the action being performed

## Demo


https://github.com/JCurl08/BitVision/assets/60952417/a963fa12-6f1d-4d5c-b083-dc1cfaac7b06



## Project Setup
1. Setup virtual environment and install dependencies according to `INSTALLATIONS.md`.
2. Install Mediapipe. Follow instructions at https://developers.google.com/mediapipe/solutions/setup_python.

## Video Recording and Data Generation
1. In DataGenerator.py, set the data file output for the specific action to be recorded on line 70.
2. In terminal, navigate to the `/predict` directory and run `python DataGenerator.py {action_class}`. This will start the webcam and begin recording mediapipe data for the specified action.
3. The output data .csv file will be stored in the `/train/training_data` directory.

## Random Forest Model Training 
1. Go to the `/train` directory.
2. Run `python ModelGenerator.py {model_name}`. This will append the action (training data file name) as the class type for the associated data and then concatenate all training data into one file.
3. The trained model will be saved to `/models` as a pickled model, `{model_name}.pkl`.

## Live Video Capture and Keyboard Input Generation
1. Go to the project root directory.
2. Run `python main.py`. This will start the webcam video capture and begin generating key inputs according to the trained model and controller inputs specified in the `Controller` module's `control_scheme`.
