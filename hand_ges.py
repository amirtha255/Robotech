import os
import cv2
#starts the webcam, uses it as video source
camera = cv2.VideoCapture(0) #uses webcam for video

while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = camera.read()
    frame_op = frame.copy()

    cv2.imshow('object detection', cv2.resize(frame_op, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()