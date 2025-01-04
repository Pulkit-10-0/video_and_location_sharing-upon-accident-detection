# Accident Detection & Emergency Response System

This project implements an automated system to detect accidents in real-time, capture critical video and audio evidence, and send emergency alerts to authorities. The system uses multiple cameras for video capture, object detection with OpenCV, and audio recording through PyAudio.

Watch a demonstration of the system in action:  
[Accident Detection System Demonstration](https://www.youtube.com/watch?v=qgTzIAcwtG4)

## Features
- **Real-Time Video Capture**: Utilizes multiple cameras to continuously monitor the vehicle’s surroundings.
- **Object Detection**: Uses a deep learning-based model in OpenCV to detect objects that may indicate an accident.
- **Audio Recording**: Records audio using a microphone, capturing crucial sounds for evidence.
- **Emergency Response**: When an accident is detected, the system:
  - Plays an alert sound.
  - Fetches the GPS location.
  - Sends an SMS with the location to emergency authorities via Twilio.

## Technologies Used
- **OpenCV**: For video capture and object detection.
- **PyAudio & SoundDevice**: For audio recording.
- **Twilio**: For sending SMS alerts to emergency responders.
- **Geocoder**: For fetching real-time GPS location.

## Installation

1. Clone the repository:
   
   `git clone https://github.com/yourusername/accident-detection.git`

   `cd accident-detection`

2. Install the required Python libraries:
   
   `pip install -r requirements.txt`

3. Download the necessary files:
   - `frozen_inference_graph.pb` (the pre-trained model)
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (the configuration file)
   - `labels.txt` (contains the class labels for object detection)

4. Replace the placeholders for Twilio credentials in the script:

   ```python
   ACCOUNT_SID = 'your_account_sid'
   AUTH_TOKEN = 'your_auth_token'
   TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
   TO_PHONE_NUMBER = 'recipient_phone_number'

## Usage 

1.Run the Python script:

```
python accident_detection.py

```

2.Press the 'a' key to simulate an accident detection. The system will then:

  - Play an alert sound.

  - Capture the current video frames and audio.

  - Send an SMS to emergency authorities with the accident location.

