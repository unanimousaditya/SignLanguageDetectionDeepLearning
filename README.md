# Sign Language Detection Deep Learning Project

This project implements a hand gesture recognition system that can detect six different hand gestures: digits ONE through FIVE and a NONE state.

## Project Structure

```
SIGNLANGUAGEDETECTIONDEE...
├── Code
│   ├── HandGestureRecognitionOpenCV.py  # Main recognition implementation
│   ├── test.py                          # Script for testing the model
│   └── TrainingHandGesture.py           # Script for training the model
├── HandGestureDataset
│   ├── test                             # Test dataset
│   │   ├── FIVE
│   │   ├── FOUR
│   │   ├── NONE
│   │   ├── ONE
│   │   ├── THREE
│   │   └── TWO
│   ├── train                            # Training dataset
│   │   ├── FIVE
│   │   ├── FOUR
│   │   ├── NONE
│   │   ├── ONE
│   │   ├── THREE
│   │   └── TWO
│   └── _DS_Store
├── .gitpod.yml
├── README.md
└── requirements.txt
```

## Features

- Real-time detection of hand gestures for digits ONE through FIVE and NONE
- Uses OpenCV for image processing
- Simple yet effective deep learning model for accurate classification
- Separate datasets for training and testing

## Requirements

To run this project, you'll need the following dependencies:
- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/unanimousaditya/MSAI-AICTE-APRIL2025-INTERNSHIP/tree/main/SignLanguageDetectionDeepLearning
   cd SignLanguageDetectionDeepLearningProject
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model on the provided dataset:

```bash
python Code/TrainingHandGesture.py
```

The script will load images from the `HandGestureDataset/train` directory, with separate subdirectories for each gesture class (ONE, TWO, THREE, FOUR, FIVE, NONE).

### Testing the Model

To evaluate the model on the test dataset:

```bash
python Code/test.py
```

This will run the trained model against images in the `HandGestureDataset/test` directory and report accuracy metrics.

### Real-time Recognition

To start real-time hand gesture recognition using your webcam:

```bash
python Code/HandGestureRecognitionOpenCV.py
```

This will open a webcam feed and begin detecting hand gestures in real-time.

## Dataset

The dataset is organized into training and testing sets, with the following classes:
- ONE (index finger)
- TWO (index and middle fingers)
- THREE (index, middle, and ring fingers)
- FOUR (all fingers except thumb)
- FIVE (all five fingers)
- NONE (no specific gesture or hand not detected)

Each class has its own directory containing the corresponding images.

## Model

The project uses a CNN (Convolutional Neural Network) to classify hand gestures. The model is trained on the dataset provided in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is available under the MIT License.

## Acknowledgments

- OpenCV for image processing capabilities
- TensorFlow/Keras for the deep learning framework