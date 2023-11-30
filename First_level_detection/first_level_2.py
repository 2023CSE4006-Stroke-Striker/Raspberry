import cv2
import dlib
import numpy as np
import time
import threading
import requests
import send_aws as send
import detect_face as detect
import voice_alert as alert

def detect_and_process_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        detect.process_face(frame, face)

# 얼굴 감지기 및 특징점 감지기 초기화
detector = dlib.get_frontal_face_detector()

# Pi camera 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print(cap.isOpened())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_process_faces(frame)
    cv2.imshow('Detect Dropped Mouth Corner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
