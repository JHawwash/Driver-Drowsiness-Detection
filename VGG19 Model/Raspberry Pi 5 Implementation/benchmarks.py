########## Importing Necessary Libraries ########## 


import cv2
import numpy as np
from pygame import mixer
from keras.models import load_model
from face_detector import YoloDetector
from picamera2 import Picamera2
import signal
import time
import psutil, subprocess
import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage


########## Initializing the Model, Alarm Audio, Camera, and Text Font ##########


# Starting PyQT Application (For Displaying the Camera Footage)
app = QApplication(sys.argv)

# Loading Our Model
drowsiness_detector_model = load_model('/home/mjr/Desktop/GP2/Transfer_VGG19/model_30.keras', compile=False)

# Loading the Pretrained Face Detector Model (yolo-face)
face_detector = YoloDetector(target_size=512, device = "cpu", min_face = 90)

# Initializing Pygame Mixer for Audio
mixer.init()
soundfile = '/home/mjr/Desktop/GP2/alarm.wav'
sound = mixer.Sound(soundfile)

# Initializing the Camera
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": 'XRGB8888', "size": (512,384)}))
cam.start()

# Font for the text
font = cv2.FONT_HERSHEY_SIMPLEX


########## Initializing Necessary Variables for Detection ##########


# Score for the number of frames where the Driver is Drowsy
drowsy_score = 0

# Threshold for the number of frames where the Driver is Drowsy
drowsy_score_threshold = 20

# Flag for the Drowsiness status, Set to True if drowsy_score exceeds the threshold drowsy_score_threshold
drowsy = False

# Counter for the number of frames where the Face is Not Detected
face_not_detected_counter = 0

# Threshold for the number of frames the Face is Not Detected
face_not_detected_threshold = 20

# Flag for the Face Status
face_not_detected = False

# Flag for the Alarm Status
alarm_playing = False

# Flag used in the Signal Handler
exit_flag = False


########## Benchmark Variables ##########


# Time taken to process each frame
frame_processing_times = []

# CPU Temperature
temperatures = []

# CPU Utilization
cpu_utilization = []

# Memory Utilization
memory_utilization = []

# GPU Utilization
gpu_utilization = []


########## FUNCTIONS ##########


# Plays the alarm indefinetely if it is not already playing
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        sound.play(-1)
        alarm_playing = True


# Stops the alarm if it is playing
def stop_alarm():
    global alarm_playing
    if alarm_playing:
        sound.stop()
        alarm_playing = False    

        
# Checks if the driver's face is visible  
def check_face(faces):
    global face_not_detected, face_not_detected_counter, face_not_detected_threshold
    
    
    if faces:
        for face in faces[0]:
            # If a face is detected, reset the face_not_detected_counter to 0
            if face:
                face_not_detected_counter = 0
            
            # If no face is detected, increment the face_not_detected_counter by 1
            elif not face:
                face_not_detected_counter += 1

    # If the face_not_detected_counter exceeds the face_not_detected_threshold, set face_not_detected to True
    if face_not_detected_counter >= face_not_detected_threshold:
        face_not_detected = True
    else:
        face_not_detected = False
        
        
# Used to exit main loop
def signal_handler(sig, frame):
    global exit_flag
    exit_flag = True


# Calculate the performance of the model on the hardware
def calc_performance():
    global frame_processing_times, temperatures, memory_utilization, gpu_utilization
    # Frame processing time
    time_diff = end_time - start_time
    frame_processing_times.append(time_diff)
    
    # Temperature
    temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
    temp = float(temp.split('=')[1].split("'")[0]) # temp=61.5'C
    temperatures.append(temp)
    
    # Memory Utilization
    mem_usage = psutil.virtual_memory().percent
    memory_utilization.append(mem_usage)
    
    # GPU Utilization
    gpu_output = subprocess.check_output(['vcgencmd', 'get_mem', 'gpu']).decode('utf-8')
    gpu_util = gpu_output.split('=')[1].strip() # gpu=4M
    gpu_util = int(''.join(filter(str.isdigit, gpu_util)))
    gpu_utilization.append(gpu_util)
    
    # CPU Utilization
    cpu_util = psutil.cpu_percent()
    cpu_utilization.append(cpu_util)
               
                                                                               
########## Drowsiness Detection Main Code ##########


# This counter is used to calculate benchmarks every 600 frames
counter = 0

# Register the signal handler (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Create a QLabel to display the camera footage
label = QLabel()

# Set the window title
label.show()

# Main Loop
while not exit_flag:
    
    # Get frame
    frame = cam.capture_array()
    
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Rotate the frame by 90 degrees counterclockwise
    # We do this since in the devices default orientation, the camera is rotated 90 degrees to the right. So we counteract this.
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Start Timer
    start_time = time.time()
    
    # Detect Faces using the Face Detector
    faces = face_detector.predict(frame)
    
    if faces:
        for face in faces[0]:
            
            # If face is not detected
            if not face:
                check_face(faces)
                
                if face_not_detected:
                    cv2.putText(frame, 'Face Not Detected', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Play alarm
                    if not alarm_playing:
                        play_alarm()
                else:
                    # Stop alarm
                    if alarm_playing:
                        stop_alarm()
                
            else:
                # Draw the rectangle around the face
                cv2.rectangle(frame, (int(face[0][0]), int(face[0][1])), (int(face[0][2]), int(face[0][3])), (0, 255, 0), 2)
                
                # Crop the face
                face = frame[int(face[0][1]):int(face[0][3]), int(face[0][0]):int(face[0][2])]
                
                # Resize the face to 64x64 (Expected by Our CNN Model)
                face = cv2.resize(face, (64, 64))
                
                # Normalize the face
                face = face/255
                
                # Expand the dimensions of the face
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)
                
                # Get the Prediction from Our CNN Model
                prediction = drowsiness_detector_model.predict(face)
                
                # If the prediction < 0.5 (Drowsy), increment the drowsy_score by 1
                if prediction < 0.5:
                    drowsy_score += 1
                    cv2.putText(frame, 'Drowsy', (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # If the prediction >= 0.5 (Awake), decrement the drowsy_score by 1
                elif prediction >= 0.5:
                    drowsy_score -= 1
                    cv2.putText(frame, 'Awake', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # If the drowsy_score < 0, set it to 0
                if drowsy_score < 0:
                    drowsy_score = 0
                
                # Display the Drowsy Score
                cv2.putText(frame,f"Drowsy Score: {drowsy_score}", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # If the drowsy_score exceeds the drowsy_score_threshold, set drowsy to True and play the alarm
                if drowsy_score >= drowsy_score_threshold:
                    drowsy = True
                    if not alarm_playing:
                        play_alarm()
                         
                # If the drowsy_score is less than the drowsy_score_threshold, set drowsy to False and stop the alarm
                else:
                    drowsy = False
                    if alarm_playing:
                        stop_alarm()
    
    
    # End Timer
    end_time = time.time()
    
    # Calculate Performance Every 100 Frames
    counter += 1
    if counter == 100:
        calc_performance()
        counter = 0
                        
    # Show the Frame    
    pixmap = QPixmap.fromImage(QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888))
    label.setPixmap(pixmap)
    app.processEvents()
    

# Close the Camera
cam.close()

# Stop the alarm if it is playing
if alarm_playing:
    stop_alarm()

# Performance Analysis
print(f"Average Frame Processing Time: {np.mean(frame_processing_times)} seconds")
print(f"Average Temperature: {np.mean(temperatures)}'C")
print(f"Average CPU Utilization: {np.mean(cpu_utilization)}%")
print(f"Average Memory Utilization: {np.mean(memory_utilization)}%")
print(f"Average GPU Utilization: {np.mean(gpu_utilization)}M")

# Writing Benchmarks to a text file
with open('benchmark_results.txt', 'w') as f:
    f.write(f"Average Frame Processing Time: {np.mean(frame_processing_times)} seconds\n")
    f.write(f"Average Temperature: {np.mean(temperatures)}'C\n")
    f.write(f"Average CPU Utilization: {np.mean(cpu_utilization)}%\n")
    f.write(f"Average Memory Utilization: {np.mean(memory_utilization)}%\n")
    f.write(f"Average GPU Utilization: {np.mean(gpu_utilization)}M\n")
    f.write("\n")
    f.write(f"frame_processing_times: {frame_processing_times}\n")
    f.write(f"temperatures: {temperatures}\n")
    f.write(f"cpu_utilizations: {cpu_utilization}\n")
    f.write(f"memory_utilizations: {memory_utilization}\n")
    f.write(f"gpu_utilizations: {gpu_utilization}\n")

# Exit the PyQT Application
sys.exit(app.exec_())                                                          
