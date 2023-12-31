import time
import cv2

def countdown(frame, flag):

    font = cv2.FONT_HERSHEY_SIMPLEX
    if(flag == 0):
        cv2.putText(frame, f'Look at the camera', (10, 100), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.waitKey(1000)
        
    if(flag == 1):
        cv2.putText(frame, f'Now Start again...', (50, 150), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.waitKey(1000)

    cv2.imshow('Detect Dropped Mouth Corner', frame)
    cv2.waitKey(1000)
    
    