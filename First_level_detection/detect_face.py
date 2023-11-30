import numpy as np
import count_clock as clock
import take_photo as photo
import send_aws as send
import voice_alert as alert
import dlib
import cv2

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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

        #show the process in voice
        if abs(lip_height) > 0.25:
            print('mvalue: ', abs(lip_height))
            alert.synthesize_and_play("Stroke symtoms has been detected. please look at the camera for 5 seconds")
            #clock.countdown(frame, 0)
            for i in range(5, 0, -1):
                print(f'Countdown: {i} seconds')
                cv2.waitKey(1000)
            photo.take_photo(frame)
            alert.synthesize_and_play("The AI model is currently analyzing the image. Please wait a moment.")
            #clock.countdown(frame, 1)
            for i in range(5, 0, -1):
                print(f'Countdown: {i} seconds')
                cv2.waitKey(1000)
            send.main("arn:aws:rekognition:ap-northeast-2:464499631690:project/stroke-nostroke-classification/version/stroke-nostroke-classification.2023-11-25T12.38.57/1700883537519", "/home/seungsu/pyvenv/Raspberry/First_level_detection/captured_image_0.jpg")
            
            
             #show the process in frame
        # if abs(lip_height) > 0.25:
        #     print('mvalue: ', abs(lip_height))
        #     cv2.putText(frame, 'Mouth Corner Dropped!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        #     clock.countdown(frame, 0)
        #     for i in range(5, 0, -1):
        #         print(f'Countdown: {i} seconds')
        #         cv2.waitKey(1000)
        #     photo.take_photo(frame)
        #     clock.countdown(frame, 1)
        #     for i in range(5, 0, -1):
        #         print(f'Countdown: {i} seconds')
        #         cv2.waitKey(1000)
        #     send.main("arn:aws:rekognition:ap-northeast-2:464499631690:project/stroke-nostroke-classification/version/stroke-nostroke-classification.2023-11-25T12.38.57/1700883537519", "/home/seungsu/pyvenv/Raspberry/First_level_detection/captured_image_0.jpg")
        