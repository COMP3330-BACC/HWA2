# Key Detector
This folder holds the script responsible for evaluating test data.

## Reliances
- The detection code relies on the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- The code also assumes that the `object_detection` folder is already within the system `PATH`

## Instructions for Operation
- The Tensorflow Object Detection API must be downloaded and added to the system `PATH`
- To firstly configure the detector, look in `config.py`. Here you can see the training image directory, preview image size and select the model (from the `models` folder) to evaluate
- The detector will look through TEST_DIR for images
- For each image found, it will display the bounding boxes
- The coordinates and corresponding confidences for the predictions will be shown in the terminal
- To progress to the next image, press any button


## Support
If there are any issues with any of these points, the code can be demonstrated in person or further instructions may be given.