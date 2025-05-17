# 🖐️ Sign Language Detection Deep Learning Project

## 📋 Overview
This project implements a comprehensive real-time hand gesture recognition system capable of detecting and classifying hand gestures representing digits ONE through FIVE, as well as a NONE state when no specific gesture is detected. The system utilizes computer vision techniques and deep learning to create an accessible tool for sign language digit recognition.

Perfect for students, researchers, and developers interested in computer vision, machine learning, and accessibility technologies. This project demonstrates how AI can be leveraged to bridge communication gaps and assist individuals who use sign language.

## 📂 Project Structure

```
SIGNLANGUAGEDETECTIONDEE...
├── Code
│   ├── HandGestureRecognitionOpenCV.py  # Main recognition implementation with OpenCV
│   ├── test.py                          # Script for evaluating model performance
│   └── TrainingHandGesture.py           # Script for training the CNN model
├── HandGestureDataset
│   ├── test                             # Test dataset for validation
│   │   ├── FIVE                         # Images of five-finger gesture
│   │   ├── FOUR                         # Images of four-finger gesture
│   │   ├── NONE                         # Images of no specific gesture
│   │   ├── ONE                          # Images of one-finger gesture
│   │   ├── THREE                        # Images of three-finger gesture
│   │   └── TWO                          # Images of two-finger gesture
│   ├── train                            # Training dataset
│   │   ├── FIVE                         # Training images of five-finger gesture
│   │   ├── FOUR                         # Training images of four-finger gesture
│   │   ├── NONE                         # Training images of no specific gesture
│   │   ├── ONE                          # Training images of one-finger gesture
│   │   ├── THREE                        # Training images of three-finger gesture
│   │   └── TWO                          # Training images of two-finger gesture
│   └── _DS_Store
├── .gitpod.yml                          # Gitpod configuration
├── README.md                            # Project documentation
└── requirements.txt                     # List of dependencies
```

## ✨ Key Features

- 🔍 **Real-time Detection**: Processes webcam feed in real-time to detect hand gestures with minimal latency
- 📷 **OpenCV Integration**: Leverages OpenCV's advanced image processing capabilities for robust hand detection
- 🧠 **Deep Learning Model**: Uses a Convolutional Neural Network (CNN) for accurate gesture classification
- 🎯 **Multi-class Classification**: Recognizes six distinct hand gestures: digits ONE through FIVE and NONE
- 🗃️ **Comprehensive Dataset**: Includes well-organized training and testing datasets for each gesture class
- 📊 **Performance Metrics**: Includes tools to evaluate and report model accuracy and performance
- 🔄 **Complete Pipeline**: Features code for data preprocessing, model training, evaluation, and deployment

## 🛠️ Technical Requirements

To run this project, you'll need the following dependencies:
- 🐍 **Python 3.x**: The primary programming language
- 📊 **OpenCV**: For image capture and processing
- 🤖 **TensorFlow/Keras**: Deep learning framework for model creation and training
- 🔢 **NumPy**: For numerical operations and array handling
- 📉 **Matplotlib**: For visualization of training progress and results (optional)
- 📦 **Additional libraries**: See requirements.txt for the complete list of dependencies

## 📥 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/unanimousaditya/SignLanguageDetectionDeepLearning.git
   cd SignLanguageDetectionDeepLearning
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your webcam is properly connected and functional for real-time detection.

## 🚀 Usage Guide

### 🏋️‍♂️ Training the Model

To train the model on the provided dataset:

```bash
python Code/TrainingHandGesture.py
```

The training script will:
- Load images from the `HandGestureDataset/train` directory
- Preprocess the images for training
- Create and train a CNN model
- Save the trained model for later use
- Display training metrics and progress

For customized training, you can modify hyperparameters in the script.

### 🧪 Testing and Evaluation

To evaluate the model's performance on the test dataset:

```bash
python Code/test.py
```

This script will:
- Load the trained model
- Run inference on images in the `HandGestureDataset/test` directory
- Calculate and report accuracy, precision, recall, and F1-score
- Generate a confusion matrix to visualize classification performance

### 📹 Real-time Recognition Demo

To start the real-time hand gesture recognition system using your webcam:

```bash
python Code/HandGestureRecognitionOpenCV.py
```

This application will:
- Access your webcam feed
- Detect hand regions in each frame
- Apply the trained model to classify detected gestures
- Display the recognized digit in real-time
- Provide visual feedback on the detected gesture

## 📊 Dataset Description

The dataset is carefully organized into training and testing sets, with the following classes:
- 1️⃣ **ONE**: Index finger extended (pointer)
- 2️⃣ **TWO**: Index and middle fingers extended (peace sign)
- 3️⃣ **THREE**: Index, middle, and ring fingers extended
- 4️⃣ **FOUR**: All fingers except thumb extended
- 5️⃣ **FIVE**: All five fingers extended (open hand)
- ❌ **NONE**: No specific gesture or hand not detected

Each class contains diverse images with variations in:
- Hand orientation and position
- Lighting conditions
- Background environments
- Hand sizes and skin tones

## 🧠 Model Architecture

The project employs a Convolutional Neural Network (CNN) architecture:
1. **Input Layer**: Accepts preprocessed hand gesture images
2. **Convolutional Layers**: Extract spatial features from images
3. **Pooling Layers**: Reduce dimensionality while preserving important features
4. **Dropout Layers**: Prevent overfitting
5. **Dense Layers**: Final classification of features into gesture classes
6. **Output Layer**: Six-node softmax layer for class probabilities

The model is optimized for both accuracy and inference speed to enable real-time detection.

## 🔮 Future Enhancements

- 🌐 Expand the recognition to include full ASL alphabet
- 📱 Mobile application deployment
- 🔄 Real-time translation of sign language sentences
- 🛠️ Improved robustness to varying lighting conditions
- 🧩 Integration with other accessibility tools

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is available under the MIT License.

## 🙏 Acknowledgments

- 📷 OpenCV community for image processing libraries
- 🤖 TensorFlow/Keras team for the deep learning framework
- 🏫 All contributors to the computer vision and sign language recognition fields
- 👥 Everyone who contributes to making technology more accessible
