########## Importing Necessary Libraries ########## 

import cv2
import numpy as np
from keras.models import load_model
from face_detector import YoloDetector
from picamera2 import Picamera2
import signal
from gpiozero import Buzzer

########## Initializing the Model, Alarm Audio, and Camera ##########


# Loading Our Model
drowsiness_detector_model = load_model('/home/mjr/Desktop/GP2/Transfer_VGG19/model_30.keras', compile=False)

# Loading the Pretrained Face Detector Model (yolo-face)
face_detector = YoloDetector(target_size=512, device = "cpu", min_face = 90)

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
drowsy_score_threshold = 3

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
    
    # Detect Faces using the Face Detector
    faces = face_detector.predict(frame)
    
    # If there are faces detected
    if faces:
        for face in faces[0]: # For the first face
            
            if face: # If the face is detected
                
                # Crop the face
                face = frame[int(face[0][1]):int(face[0][3]), int(face[0][0]):int(face[0][2])]
                
                # Convert face to grayscale
                gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                
                # Apply CLAHE to face
                clahe_face = apply_clahe(gray_face)
                
                # Apply Gamma Correction to face
                brightened_face = adjust_gamma(clahe_face, gamma=1.5)
                
                # Convert the frame back to RGB
                brightened_face = cv2.cvtColor(brightened_face, cv2.COLOR_GRAY2RGB)
                
                # Resize the face to 64x64 (Expected by Our CNN Model)
                face_resized = cv2.resize(brightened_face, (64, 64))
                
                # Normalize the face
                face_resized = face_resized/255
                
                # Expand the dimensions of the face
                face_resized = np.expand_dims(face_resized, axis=0)
                
                # Get the Prediction from Our CNN Model
                prediction = drowsiness_detector_model.predict(face_resized)
                
                # If the prediction < 0.5 (Drowsy), increment the drowsy_score by 1
                if prediction < 0.5:
                    drowsy_score += 1
                                    
                # If the prediction >= 0.5 (Awake), decrement the drowsy_score by 1
                elif prediction >= 0.5:
                    drowsy_score = 0
    
                
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
                    
            else: # If the face is not detected
                drowsy_score = 0 # Reset drowsy score to 0
                # Stop the alarm if its playing
                if alarm_playing: 
                        stop_alarm()
                        
    

# Close the Camera
cam.close()

# Stop the alarm if it is playing
if alarm_playing:
    stop_alarm()                                                     
