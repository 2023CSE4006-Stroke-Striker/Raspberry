import cv2

def take_photo(frame):
    global image_count
    # filename = f'captured_image_{image_count}.jpg'
    filename = f'captured_image_0.jpg'
    cv2.imwrite(filename, frame)
    print(f'Photo saved as {filename}')
    
    