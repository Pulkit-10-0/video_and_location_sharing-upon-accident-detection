import cv2
import pyaudio
import wave
import threading
from collections import deque
from moviepy.editor import VideoFileClip, AudioFileClip
import keyboard
import sounddevice as sd
import soundfile as sf
import time
from twilio.rest import Client
import geocoder

# Twilio credentials

# enter crendentials for sms service 

ACCOUNT_SID = 'xxxxxxxxxxxxxxxxxxxxxxx'
AUTH_TOKEN = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'
TWILIO_PHONE_NUMBER = 'xxxxxxxxxxxx'
TO_PHONE_NUMBER = 'xxxxxxxx'

def play_beep():
    try:
        data, samplerate = sf.read("security-alarm-63578.mp3")  
        sd.play(data, samplerate)
        sd.wait() 
    except Exception as e:
        print(f"Failed to play beep sound: {e}")

def send_sms_to_authorities(location):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body=f"Emergency: Car crash detected at location: {location}. Please send help immediately.",
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"Signal sent to authorities via SMS. Message SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def get_current_location():
    try:
        g = geocoder.ip('me') 
        if g.ok:
            location = f"Latitude: {g.latlng[0]}, Longitude: {g.latlng[1]}"
            return location
        else:
            return "Location unavailable"
    except Exception as e:
        print(f"Error fetching location: {e}")
        return "Location unavailable"

# Parameters for video
video_duration = 30  # Seconds
fps1 = 10 # Frames per second
fps2 = 10 # Frames per second
fps3 = 10 # Frames per second
video_width1 = 1280
video_height1 = 720
video_width2 = 1280
video_height2 = 720
video_width3 = 640
video_height3 = 480
buffer_size1 = video_duration * fps1  # Number of frames to store in memory
buffer_size2 = video_duration * fps2 # Number of frames to store in memory
buffer_size3 = video_duration * fps3 # Number of frames to store in memory

# Initialize deques to store frames for all three cameras
frame_buffer_1 = deque(maxlen=buffer_size1)
frame_buffer_2 = deque(maxlen=buffer_size2)
frame_buffer_3 = deque(maxlen=buffer_size3)

# Audio settings
audio_format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
audio_output_file = "accident_audio.wav"


audio = pyaudio.PyAudio()

# Function to capture audio
def record_audio():
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    frames = []
    
    print("Recording audio...")
    
    while recording:
        data = stream.read(chunk)
        frames.append(data)
    
    print("Audio recording stopped.")
    

    wf = wave.open(audio_output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


cap1 = cv2.VideoCapture(0)  
cap2 = cv2.VideoCapture(1)  
cap3 = cv2.VideoCapture(2)  

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, video_width1)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height1)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, video_width2)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height2)

cap3.set(cv2.CAP_PROP_FRAME_WIDTH, video_width3)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height3)

# Object Detection 
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(len(classLabels))

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

recording = True

# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

print("Recording video... Press 'a' to simulate accident detection.")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    
    if not ret1 or not ret2 or not ret3:
        break
    

    ClassIndex1, confidence1, bbox1 = model.detect(frame1, confThreshold=0.55)
    if len(ClassIndex1) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex1.flatten(), confidence1.flatten(), bbox1):
            if ClassInd <= 80:
                cv2.rectangle(frame1, boxes, color=(255, 0, 0), thickness=2)
                cv2.putText(frame1, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, color=(0, 255, 0), thickness=3)
    

    ClassIndex2, confidence2, bbox2 = model.detect(frame2, confThreshold=0.55)
    if len(ClassIndex2) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex2.flatten(), confidence2.flatten(), bbox2):
            if ClassInd <= 80:
                cv2.rectangle(frame2, boxes, color=(255, 0, 0), thickness=2)
                cv2.putText(frame2, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, color=(0, 255, 0), thickness=3)
    

    ClassIndex3, confidence3, bbox3 = model.detect(frame3, confThreshold=0.55)
    if len(ClassIndex3) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex3.flatten(), confidence3.flatten(), bbox3):
            if ClassInd <= 80:
                cv2.rectangle(frame3, boxes, color=(255, 0, 0), thickness=2)
                cv2.putText(frame3, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, color=(0, 255, 0), thickness=3)
    

    frame_buffer_1.append(frame1)
    frame_buffer_2.append(frame2)
    frame_buffer_3.append(frame3)
    

    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    cv2.imshow('Camera 3', frame3)
    
    # Check for 'a' key press to simulate accident detection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        print("Accident detected. Stopping recording...")
        recording = False
        play_beep()
        location = get_current_location()
        print(f"Current location: {location}")
        send_sms_to_authorities(location)
        time.sleep(2)  
        break  


cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()

audio_thread.join()

video_output_file_1 = "accident_clip_camera_1.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(video_output_file_1, fourcc, fps1, (video_width1, video_height1))

for frame in frame_buffer_1:
    out1.write(frame)

out1.release()
print(f"Video from camera 1 saved as {video_output_file_1}")

video_output_file_2 = "accident_clip_camera_2.avi"
out2 = cv2.VideoWriter(video_output_file_2, fourcc, fps2, (video_width2, video_height2))

for frame in frame_buffer_2:
    out2.write(frame)

out2.release()
print(f"Video from camera 2 saved as {video_output_file_2}")

video_output_file_3 = "accident_clip_camera_3.avi"
out3 = cv2.VideoWriter(video_output_file_3, fourcc, fps3, (video_width3, video_height3))

for frame in frame_buffer_3:
    out3.write(frame)

out3.release()
print(f"Video from camera 3 saved as {video_output_file_3}")
audio.terminate()

# Combine video from the first camera with audio
video_clip_1 = VideoFileClip(video_output_file_1)
audio_clip = AudioFileClip(audio_output_file)


clip_duration = min(video_clip_1.duration, audio_clip.duration, 30)


if video_clip_1.duration > clip_duration:
    video_clip_1 = video_clip_1.subclip(max(0, video_clip_1.duration - clip_duration), video_clip_1.duration)
if audio_clip.duration > clip_duration:
    audio_clip = audio_clip.subclip(max(0, audio_clip.duration - clip_duration), audio_clip.duration)


final_clip_1 = video_clip_1.set_audio(audio_clip)
final_output_file_1 = "1st_cam_accident_with_audio.mp4"  # Updated file name
final_clip_1.write_videofile(final_output_file_1, codec="libx264", audio_codec="aac")

print(f"Final video with audio saved as {final_output_file_1}")


final_output_file_2 = "2nd_cam_accident_with_audio.mp4"
video_clip_2 = VideoFileClip(video_output_file_2)
video_clip_2.write_videofile(final_output_file_2, codec="libx264")

print(f"Second camera video saved as {final_output_file_2}")


final_output_file_3 = "3rd_cam_accident_without_audio.mp4"
video_clip_3 = VideoFileClip(video_output_file_3)
video_clip_3.write_videofile(final_output_file_3, codec="libx264")

print(f"Third camera video saved as {final_output_file_3}")
