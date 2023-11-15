import cv2
import dlib
import numpy as np
import time
from imutils import face_utils


# !wget   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 # DOWNLOAD LINK

# !bunzip2 /content/shape_predictor_68_face_landmarks.dat.bz2

# datFile =  "/content/shape_predictor_68_face_landmarks.dat"
# 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()
print("after detector")

# 얼굴 특징점 감지기 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 노트북 웹캠 열기
cap = cv2.VideoCapture(0)

image_count = 0

def take_photo(frame):
    global image_count
    filename = f'captured_image_{image_count}.jpg'
    cv2.imwrite(filename, frame)
    print(f'Photo saved as {filename}')
    image_count += 1
    
def countdown(frame, flag):
    center_coordinates = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    for i in range(3, 0, -1):
        print(f'Countdown: {i} seconds')
        time.sleep(1)
        frame = np.zeros_like(frame)

        # 프레임에 카운트 다운 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        ## before take_photo 
        if(flag == 0):
            cv2.putText(frame, f'Detect Dropped Mouth', (center_coordinates[0]-300, center_coordinates[1]-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Countdown: {i} seconds, Look at the camera', (center_coordinates[0]-300, center_coordinates[1]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Detect Dropped Mouth Corner', frame)
            cv2.waitKey(1000)  # 1초 대기
            
        ## after take_photo
        if(flag == 1):
            cv2.putText(frame, f'Start again after {i} seconds', (center_coordinates[0]-200, center_coordinates[1]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Detect Dropped Mouth Corner', frame)
            cv2.waitKey(1000)  # 1초 대기

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 얼굴의 각도 계산
        angle = np.arctan2(landmarks.part(27).y - landmarks.part(8).y, landmarks.part(27).x - landmarks.part(8).x)
        angle = np.degrees(angle)
        
        print("angle: ", angle)
        if(85 <= abs(angle) <= 95):

            # 아랫 입술의 중심 좌표 추출
            lower_lip_center = (landmarks.part(57).x, landmarks.part(57).y)

            # 양쪽 입꼬리 좌표 추출
            left_mouth_corner = (landmarks.part(48).x, landmarks.part(48).y)
            right_mouth_corner = (landmarks.part(54).x, landmarks.part(54).y)

            # 입술 가장 아랫부분을 기준으로 양쪽 입꼬리의 상대 좌표 계산
            relative_left_corner = (left_mouth_corner[1] - lower_lip_center[1]) / (left_mouth_corner[0] - lower_lip_center[0])
            relative_right_corner = (right_mouth_corner[1] - lower_lip_center[1]) / (right_mouth_corner[0] - lower_lip_center[0])

            # 입술의 높이 계산
            lip_height = abs(relative_left_corner) - abs(relative_right_corner)

            # 내려간 입꼬리 감지
            if abs(lip_height) > 0.2:
                cv2.putText(frame, 'Mouth Corner Dropped!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                countdown(frame, 0)
                take_photo(frame)
                countdown(frame, 1)
                continue
                
            #print("lib_height: ", abs(lip_height))

            # 결과 표시
            cv2.circle(frame, lower_lip_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, left_mouth_corner, 2, (0, 255, 0), -1)
            cv2.circle(frame, right_mouth_corner, 2, (0, 255, 0), -1)

            # 양쪽 입꼬리의 상대 좌표 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Left Corner: {relative_left_corner}', (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Right Corner: {relative_right_corner}', (10, 90), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detect Dropped Mouth Corner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
