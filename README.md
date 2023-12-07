# Emotion Detector Project

This project demonstrates an emotion detector using deep learning and computer vision. The goal is to identify emotions in facial expressions using a pre-trained deep learning model.

## Dependencies

Make sure you have the following Python libraries installed:

- TensorFlow
- OpenCV (cv2)
- Matplotlib
- NumPy

You can install these dependencies using the following commands:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

### Dataset

The project uses a custom dataset stored in the "dataset" folder. The dataset includes images categorized into seven classes: "0" (Angry), "1" (Disgust), "2" (Fear), "3" (Happy), "4" (Neutral), "5" (Sad), and "6" (Surprise).

### Data Preprocessing

The notebook reads images from the dataset, resizes them to a specified dimension (224x224), and normalizes the pixel values to be in the range [0, 1]. The images are then shuffled, and the features and labels are extracted.

### Transfer Learning - Model Creation

The project utilizes transfer learning with the MobileNetV2 pre-trained model. Additional layers are added for fine-tuning the model to the emotion classification task. The model is compiled using the sparse categorical crossentropy loss function and the Adam optimizer.

### Model Training

The notebook trains the model on the preprocessed dataset for 25 epochs. The trained model is saved as "emotion_detector_model.h5" for future use.

### Emotion Detection in Real-time

The notebook captures video frames from the webcam and uses a Haar Cascade classifier to detect faces. For each detected face, the model predicts the emotion (Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise) and overlays the result on the video feed. The emotion is displayed as text and a colored rectangle around the detected face.
