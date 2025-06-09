# Sign Language Detection using OpenCV, MediaPipe, TensorFlow, and Streamlit

#### Video Demo:  [URL](https://youtu.be/Xsgc5DUmb4s)
#### Description:
This project implements a real-time sign language detection model using OpenCV, MediaPipe, TensorFlow, and Streamlit. The primary goal is to enhance communication accessibility for the specially-abled. The project incorporates various technologies to achieve this, including hand tracking, landmark extraction, and LSTM-based deep learning for gesture recognition.

## Key Components

### Model Architecture

The deep learning model is a Sequential neural network with LSTM layers for capturing temporal dependencies. It is trained to recognize specific sign language gestures related to common actions like 'hello', 'thanks', and 'iloveyou'. The model architecture includes LSTM layers followed by densely connected layers.

### Real-time Detection

The project uses OpenCV for real-time video stream processing and MediaPipe for accurate hand tracking and landmark extraction. The detection process involves feeding the extracted landmarks into the trained LSTM model to predict and interpret sign language gestures.

### Streamlit Interface

The user interface is built using Streamlit, a web application framework for creating interactive dashboards with minimal code. The interface includes buttons to run and stop the model, displaying real-time video stream processing with detected landmarks.

### Speech Feedback

The application provides speech feedback using the `pyttsx3` library, allowing the system to verbally communicate the recognized sign language gestures.

## How to Run

1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `streamlit run sign.py`.

## Usage

- Click the "Run" button to start the sign language detection model.
- The real-time video stream will display with overlaid landmarks.
- The detected sign language gestures are interpreted and displayed.
- Click the "Stop" button to end the detection process.

## Acknowledgments

Special thanks to the contributors of OpenCV, MediaPipe, TensorFlow, and Streamlit for providing the essential tools and libraries for the successful implementation of this sign language detection project.

---
