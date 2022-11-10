# Perform background subtraction
import cv2
import numpy as np

'''
# source: https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/#:~:
text=Read%2C%20Write%20and%20Display%20a%20video%20using%20OpenCV,3%20Writing%20a%20video%20...%204%20Summary%20
'''
def display_video(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
    
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    
    cap.release()
    cv2.destroyAllWindows()

'''
source: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
'''
def background_subtract(path):
    backSub = cv2.createBackgroundSubtractorKNN()
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(path))
    if not capture.isOpened():
        print('Unable to open: ' + path)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        fgMask = backSub.apply(frame)
        fgMask = cv2.medianBlur(fgMask, 5)
        fgMask = image_enhancement(fgMask)
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def image_enhancement(frame):
    err_kernel = np.ones((5, 5), np.uint8)
    dil_kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(frame, err_kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, dil_kernel, iterations=1)
    _, img_threshold = cv2.threshold(img_dilation, 70, 255, cv2.THRESH_BINARY)
    return img_threshold

if __name__ == "__main__":
    path = 'C:\\Users\\LyuxingHe\\Desktop\\GitRepos\\ece549\\BFHI\\input\\person15_running_d1_uncomp.avi'
    background_subtract(path)