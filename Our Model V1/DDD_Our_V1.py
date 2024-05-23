########## Importing Necessary Libraries ##########

import sys
import numpy as np
from keras.models import load_model
from picamera2 import Picamera2
import cv2
import signal
from gpiozero import Buzzer
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage

########## Initializing the Model, Cascades, Buzzer, Camera, Output Window, and Text Font ##########

# Start PyQT Application
app = QApplication(sys.argv)

# Loading the Model
model = load_model('/home/mjr/Desktop/GP2/Our_Model_V1/model.h5')

# Initializing Left Eye, Right Eye, and Face Cascades from OpenCV
face_classifier = cv2.CascadeClassifier('/home/mjr/Desktop/GP2/Our_Model_V1/haarcascade/haarcascade_frontalface_alt.xml')
left_eye_classifier = cv2.CascadeClassifier('/home/mjr/Desktop/GP2/Our_Model_V1/haarcascade/haarcascade_lefteye_2splits.xml')
right_eye_classifier = cv2.CascadeClassifier('/home/mjr/Desktop/GP2/Our_Model_V1/haarcascade/haarcascade_righteye_2splits.xml')

# Initializing the GPIO pins and Buzzer (For the Alarm)
buzzer = Buzzer(17)

# Initializing the Camera
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": 'XRGB8888', "size": (512, 384)}))
cam.start()

# Font for the text
font = cv2.FONT_HERSHEY_SIMPLEX


########## Initializing Necessary Variables for Detection ##########

# Score for the number of frames the eyes are closed
eye_score = 0

# Threshold for the number of frames the eyes are closed
eye_score_threshold = 6

# Flag for the Eye Status
eyes_closed = False

# Flag for the Alarm Status
alarm_playing = False

# Flag to indicate if 'q' is pressed to exit the loop
exit_flag = False


########## FUNCTIONS ##########

# Plays the alarm indefinitely if it is not already playing
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


# Used to exit the main loop
def signal_handler(sig, frame):

    global exit_flag

    exit_flag = True

########## Main Loop ##########

# Register the signal handler (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Create a QLabel to display the camera footage
label = QLabel()
label.show()

while not exit_flag:

    # Capture a frame from the camera
    frame = cam.capture_array()
    
    # Rotate the frame by 90 degrees, since the camera is mounted 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Frame height, Frame width
    fh, fw = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the Face
    faces = face_classifier.detectMultiScale(gray, minNeighbors=20, scaleFactor=1.1, minSize=(25, 25))

    # Check if the Face is detected
    if len(faces) != 0:

        for (x, y, w, h) in faces:
            # Draw a Rectangle around the Face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Detect the Left Eye
        left_eye = left_eye_classifier.detectMultiScale(gray)
        
        # Detect the Right Eye
        right_eye = right_eye_classifier.detectMultiScale(gray)

        lpred, rpred = 1, 1  # Initialize predictions to 'open'


        for (x, y, w, h) in left_eye:

            eye = frame[y:y + h, x:x + w] # crop the eye

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) # draw a rectangle around the eye

            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) # convert to grayscale

            eye = cv2.resize(eye, (64, 64)) # resize to 64x64 (size expected by the model)

            eye = eye / 255 # normalize the pixel values

            eye = eye.reshape(64, 64, -1) # reshape

            eye = np.expand_dims(eye, 0) # expand dimensions

            lpred = model.predict(eye) # get prediction


        for (x, y, w, h) in right_eye:

            eye = frame[y:y + h, x:x + w] # crop the eye

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) # draw a rectangle around the eye

            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) # convert to grayscale

            eye = cv2.resize(eye, (64, 64)) # resize to 64x64 (size expected by the model)

            eye = eye / 255 # normalize the pixel values 

            eye = eye.reshape(64, 64, -1) # reshape

            eye = np.expand_dims(eye, 0) # expand dimensions

            rpred = model.predict(eye) # get prediction


        # if the Eyes are Closed(0)
        if lpred <= 0.5 or rpred <= 0.5:

            eye_score += 1

            cv2.putText(frame, "Eyes Closed", (10, fh - 150), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        else: # open (1)

            eye_score = 0

            cv2.putText(frame, "Eyes Open", (10, fh - 150), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the Eyes Score
        cv2.putText(frame, f"Eyes Score: {eye_score}", (10, fh - 100), font, 1, (255, 255, 0), 1, cv2.LINE_AA)

        

        # if the score exceeds the threshold, play the alarm
        if eye_score > eye_score_threshold:
            eyes_closed = True
            if not alarm_playing:
                play_alarm()

        # else, stop the alarm
        else:
            eyes_closed = False
            if alarm_playing:
                stop_alarm()

    # Show the Frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixmap = QPixmap.fromImage(QImage(rgb_frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888))
    label.setPixmap(pixmap)
    app.processEvents()

# Release the Camera and Destroy the Windows
cam.close()

# Stop the alarm if it is playing
if alarm_playing:
    stop_alarm()

# Exit the PyQT Application

sys.exit(app.exec_()) 

