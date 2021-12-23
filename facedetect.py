import cv2
import RPi.GPIO as GPIO
    # Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # To capture video from webcam.
cap = cv2.VideoCapture(0)
    # To use a video file as input
while True:
        # Read the frame
    _, img = cap.read()

        # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display
    cv2.imshow('img', img)
    # To avtivate the current in GPIO pin
    if (len(faces))==0:
        GPIO.output(7, False)
    elif (len(faces))==1:
        GPIO.output(7,True)

        # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Release the VideoCapture object
cap.release()