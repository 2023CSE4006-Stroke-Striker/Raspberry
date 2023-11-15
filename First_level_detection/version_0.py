import cv2
import dlib
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

        # 입술 랜드마크 인덱스: 48-68
        mouth_landmarks = landmarks.parts()[48:68]

        # 입술 중앙 위치 계산
        #mouth_center = (mouth_landmarks[6].x, mouth_landmarks[6].y)

        # 입술 중앙 지점을 표시 (선택 사항)
        #cv2.circle(frame, mouth_center, 3, (0, 0, 255), -1)

        # 입술 중앙 위치가 얼굴 중앙보다 아래에 있으면 드루핑으로 판단
        # if mouth_center[1] > face.top():
        #     cv2.putText(frame, "Face Drooping Detected", (face.left(), face.top() - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #입술의 바텀 기준 왼쪽과 오른쪽의 m값 차이의 평균
        left_mvalue = (landmarks.part(58).y - landmarks.part(49).y) / (landmarks.part(58).x - landmarks.part(49).x)
        right_mvalue = (landmarks.part(58).y - landmarks.part(55).y) / (landmarks.part(58).x - landmarks.part(55).x)
        mvalue = abs(left_mvalue) - abs(right_mvalue)
        print("mvalue: ", mvalue)
        
        if abs(mvalue) > 1.2:
            cv2.putText(frame, "Face Drooping Detected", (face.left(), face.top()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    rects = detector(gray, 0)
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
      
      
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    cv2.imshow("Face Drooping Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
