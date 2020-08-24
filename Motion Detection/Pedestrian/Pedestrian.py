import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    #use cv2.absdiff of frame1 and frame2
    diff = cv2.absdiff(frame1, frame2)
    #use cv2.cvtColor on diff and convert BGR2GRAY
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #use cv2.GaussianBlur on gray
    blur = cv2.GaussianBlur(gray ,(3,3), cv2.BORDER_DEFAULT)  #not sure about this 3
    cv2.imshow("blur", blur)
    #use cv2.threshold on blur
    _, thresh = cv2.threshold(blur,20,255,cv2.cv2.THRESH_BINARY)

    cv2.imshow("thresh", thresh)

    #use cv2.dilate suggested iteration = 3
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 3)

    # cv2.imshow("dilated", dilated)

    #use cv2.findContours on dilated image, use cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE as parameters
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #create bounding rect on contour using cv2.boundingRect
        (x, y, w, h) = cv2.boundingRect(contour)

        # if cv2.contourArea(contour) < 900 or cv2.contourArea(contour) > 3000:
        if cv2.contourArea(contour) < 8000:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # exclude contours that are not in the estimated area size
    new_contours = []
    for contour in contours:
        #if cv2.contourArea(contour) >900 and cv2.contourArea(contour) < 3000:
        if cv2.contourArea(contour) > 8000:
            new_contours.append(contour)
    
    cv2.drawContours(frame1, new_contours, -1, (0, 0, 255), 2)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()