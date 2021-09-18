import cv2
import numpy as np

################# Video ###############

vid = cv2.VideoCapture("assets/earth.mp4")
while True:
    success,img = vid.read()
    cv2.imshow("Video Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


########## Webcam ##############

webcam = cv2.VideoCapture(1)
webcam.set(3,640)  # width
webcam.set(4,480)  # height
webcam.set(10,100)  # brightness

while True:
    success,img = webcam.read()
    cv2.imshow("Webcam Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break