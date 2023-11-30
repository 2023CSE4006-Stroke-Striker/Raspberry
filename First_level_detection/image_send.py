import requests
# import google.generativeai as palm  (pip install google-generativeai)
# from gtts import gTTS            (pip install gTTS)
# from API_KEY import API_KEY
# from playsound import playsound (pip install playsound)

def image_send(): 
    # FastAPI 서버의 엔드포인트 URL
    server_url = "http://3.34.6.216:8000/classify"

    # 업로드할 이미지 파일 경로
    image_path = "/home/seungsu/pyvenv/FaceDetection/captured_image_0.jpg"

    # 이미지 파일을 업로드할 때 사용할 파일 이름
    file_name = "uploaded_image.jpg"

    # 파일 업로드를 위한 데이터 준비
    files = {"file": (file_name, open(image_path, "rb"), "image/jpeg")}

    # POST 요청 보내기
    response = requests.post(server_url, files=files)

    # JSON 응답을 파이썬 객체로 변환
    json_response = response.json()

    # 각 항목 출력
    prediction = json_response['prediction']
    probability_0 = round(json_response['probability_0'], 2)
    probability_1 = round(json_response['probability_1'], 2)

    # 응답 확인

    if prediction == 0:
        print("Stroke is not detected !")
        print("Accuracy: ", probability_0)
        return 1
    else:
        print("Stroke is detected !")
        print("Accuracy: ", probability_1)
        '''
        palm.configure(api_key=API_KEY)
        prompt = "say to someone that he is having a stroke"  # we can optimize this by selecting the best prompt for our scenario
        response = palm.generate_text(prompt=prompt)
        response = response.result.replace('*', ' ') 
        print(response)
        tts = gTTS(response)
        tts.save('speech.mp3')
        playsound("speech.mp3")
        '''
        return 1
