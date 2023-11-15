import cv2
import dlib
import numpy as np
import time
import threading
import requests
import image_send as send

# 얼굴 감지기 초기화
image_count = 0

def take_photo(frame):
    global image_count
    filename = f'captured_image_{image_count}.jpg'
    cv2.imwrite(filename, frame)
    print(f'Photo saved as {filename}')
    image_count += 1
    
def countdown(frame, flag):
    
    # for i in range(5, 0, -1):
    #     print(f'Countdown: {i} seconds')
        
        # 프레임에 카운트 다운 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    if(flag == 0):
        cv2.putText(frame, f'Look at the camera', (10, 100), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.waitKey(1000)
        
    if(flag == 1):
        cv2.putText(frame, f'Now Start again...', (50, 150), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.waitKey(1000)

    cv2.imshow('Detect Dropped Mouth Corner', frame)
    cv2.waitKey(1000)

def process_face(frame, face):
    landmarks = predictor(frame, face)
    
    # 얼굴의 각도 계산
    angle = np.arctan2(landmarks.part(27).y - landmarks.part(8).y, landmarks.part(27).x - landmarks.part(8).x)
    angle = np.degrees(angle)
    
    print("angle: ", angle)
    
    if 85 <= abs(angle) <= 95:
        lower_lip_center = (landmarks.part(57).x, landmarks.part(57).y)
        left_mouth_corner = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth_corner = (landmarks.part(54).x, landmarks.part(54).y)
        relative_left_corner = (left_mouth_corner[1] - lower_lip_center[1]) / (left_mouth_corner[0] - lower_lip_center[0])
        relative_right_corner = (right_mouth_corner[1] - lower_lip_center[1]) / (right_mouth_corner[0] - lower_lip_center[0])
        lip_height = abs(relative_left_corner) - abs(relative_right_corner)
        
        cv2.circle(frame, lower_lip_center, 2, (0, 255, 0), -1)
        cv2.circle(frame, left_mouth_corner, 2, (0, 255, 0), -1)
        cv2.circle(frame, right_mouth_corner, 2, (0, 255, 0), -1)

        if abs(lip_height) > 0.2:
            print('mvalue: ', abs(lip_height))
            cv2.putText(frame, 'Mouth Corner Dropped!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            countdown(frame, 0)
            for i in range(5, 0, -1):
                print(f'Countdown: {i} seconds')
                cv2.waitKey(1000)
            take_photo(frame)
            countdown(frame, 1)
            for i in range(5, 0, -1):
                print(f'Countdown: {i} seconds')
                cv2.waitKey(1000)
            send.image_send()

def detect_and_process_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        process_face(frame, face)

# 얼굴 감지기 및 특징점 감지기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 노트북 웹캠 열기
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
