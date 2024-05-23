########## Importing Necessary Libraries ##########
import cv2
import numpy as np
import torch
from torch.hub import load
from picamera2 import Picamera2
import signal
from gpiozero import Buzzer

# Fixing the Pathlib Issue (Changing Windows Path to Posix Path)
import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

########## Initializing the Model, Buzzer, and Camera ##########


# Loading Our YOLO Model
drowsiness_detector_model = torch.hub.load('ultralytics/yolov5','custom',path='/home/mjr/Desktop/GP2/YOLO/yolov5/runs/train/exp5/weights/best.pt',force_reload=True)

# Initializing the GPIO pins and Buzzer (For the Alarm)
buzzer = Buzzer(17)

# Initializing the Camera
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": 'XRGB8888', "size": (512,384)}))
cam.start()


########## Initializing Necessary Variables for Detection ##########


# Score for the number of frames where the Driver is Drowsy
drowsy_score = 0

# Threshold for the number of frames where the Driver is Drowsy
drowsy_score_threshold = 2

# Flag for the Drowsiness status, Set to True if drowsy_score exceeds the threshold drowsy_score_threshold
drowsy = False

# Flag for the Alarm Status
alarm_playing = False

# Flag used in the Signal Handler
exit_flag = False

########## FUNCTIONS ##########


# Plays the alarm indefinetely if it is not already playing
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        buzzer.on()
        alarm_playing = True

# Stops the alarm if it is playing
def stop_alarm():
    global alarm_playing
    if alarm_playing:
        buzzer.off()
        alarm_playing = False                                                                             


# Used to exit main loop
def signal_handler(sig, frame):
    global exit_flag
    exit_flag = True


# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Create a CLAHE object
    return clahe.apply(image) # Apply CLAHE to the image

# Apply Gamma Correction (Brightness Adjustment)
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma # Calculate the inverse of the gamma 
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8") # Create a lookup table
    return cv2.LUT(image, table)  # Apply the lookup table to the image
    
########## Drowsiness Detection Main Code ##########

# Register the signal handler (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Main Loop
while not exit_flag:
    
    # Get frame
    frame = cam.capture_array()
    
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Rotate the frame by 90 degrees counterclockwise
    # We do this since in the devices default orientation, the camera is rotated 90 degrees to the right. So we counteract this.
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert frame to grayscale
    clahe_frame = apply_clahe(gray_frame) # apply CLAHE 
    brightened_frame = adjust_gamma(clahe_frame, gamma=1.5) # apply Gamma Correction
    corrected_frame = cv2.cvtColor(brightened_frame, cv2.COLOR_GRAY2RGB) # convert frame back to RGB
    
    # Detecting the Face
    prediction = drowsiness_detector_model(frame)
    
    if prediction:        
        # if prediction.xyxy[0].shape[0] != 0: # If detection is present (face is detected)
            
        if 16 in prediction.xyxy[0][:, 5]:  # If Drowsy (class 16)
            drowsy_score += 1  # Increment score
            
        elif 15 in prediction.xyxy[0][:, 5]:  # If Awake (class 15)
            drowsy_score = 0  # Reset score
    
                
        if drowsy_score >= drowsy_score_threshold:
            drowsy = True
            if not alarm_playing:
                play_alarm()
        
        else:
            drowsy = False
            if alarm_playing:
                stop_alarm()
        
    # Render the frame
    rendered_frame = np.squeeze(prediction.render())
            

# Close the Camera
cam.close()

# Stop the Alarm if it is playing
if alarm_playing:
    stop_alarm()       
