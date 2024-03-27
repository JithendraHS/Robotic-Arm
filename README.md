# Object Detection and Robotic Arm Control

This program detects colored objects in a video stream using OpenCV and performs basic inverse kinematics calculations to control a robotic arm. The code is written in Python and requires various libraries including OpenCV, NumPy, speech_recognition, and win32com.

## Installation and Dependencies

Ensure you have Python installed along with the required libraries. You can install them via pip:

```bash
pip install numpy opencv-python-headless imutils SpeechRecognition pywin32
```

Make sure you have a compatible webcam connected to your system. Additionally, adjust the COM port and other hardware configurations according to your setup.

## Usage

1. Run the code in a Python environment.
2. The program will prompt you to choose a colored object (e.g., red, green, blue) by voice command. Speak out the desired color.
3. Once the color is selected, the program will start detecting objects of that color in the video stream.
4. When an object is detected, it will be outlined with a colored rectangle and labeled with its color.
5. The program will capture an image of the detected object and perform further processing.
6. Inverse kinematics calculations are performed to determine the angles required for the robotic arm to pick up the object.
7. The angles are sent to the robotic arm via the specified COM port.
8. Press 'q' to quit the program.

## Files Included

- `README.md`: This readme file providing an overview of the code and instructions.
- `main.py`: The main Python script containing the object detection and robotic arm control code.

## Credits

This code was developed by Jithendra HS.
