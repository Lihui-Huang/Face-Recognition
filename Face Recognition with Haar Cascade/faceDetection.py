#import libraries
import cv2
import numpy as np

# import Classifier for Face and Eye Detection use cv2.CascadeClassifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
right_eyes_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
left_eyes_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def face_detector (img, size=0.5):
    # Convert Image to Grayscale using cv2.cvtColor
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #use face_classifier.detectMultiScale and use gray_img, set scaleFactor, minNeighbors, minSize, flags
    faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE) # minSize=(150, 100), 
    if faces is ():
        return img

    # Given coordinates to detect face and eyes location from ROI
    for (x, y, w, h) in faces:
        #x = x - 100
        #w = w + 100
        #y = y - 100
        #h = h + 100
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # blue

        # detect right eye
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #use eye_classifier.detectMultiScale and use roi_gray
        right_eyes = right_eyes_classifier.detectMultiScale(roi_gray, minSize=(50, 50)) # minSize=(50, 50)

        for (ex, ey, ew, eh) in right_eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)  #red
            roi_color = cv2.flip (roi_color, 1)

        
        # detect left eye
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #use eye_classifier.detectMultiScale and use roi_gray
        left_eyes = left_eyes_classifier.detectMultiScale(roi_gray, minSize=(50, 50)) # minSize=(50, 50)

        for (ex, ey, ew, eh) in left_eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2) #black
            roi_color = cv2.flip (roi_color, 1)

        # detect a smile in the lower half of the face
        lower_gray = gray_img[(y + h//2):y+h , x:x+w]
        roi_color = img[(y + h//2): y+h, x:x+w]

        smile = smile_classifier.detectMultiScale(lower_gray, minSize=(100, 100))

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2) #green
            roi_color = cv2.flip (roi_color, 1)

    return img

# Webcam setup for Face Detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Our Face Extractor", face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key to end face detection
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()