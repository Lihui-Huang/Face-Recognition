import cv2
import numpy as np

sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

cap = cv2.VideoCapture('pedestrian.mp4')
_, frame1 = cap.read()
_, frame2 = cap.read()

while(True):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)   

    #calculate distance between 2 frames using distMap
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing using cv2.GaussianBlur and dist mapping
    mod = cv2.GaussianBlur(dist ,(3,3), cv2.BORDER_DEFAULT)

    # apply thresholding using cv2.threshold and mod
    _, thresh = cv2.threshold(mod,20,255,cv2.cv2.THRESH_BINARY)

    # calculate std dev on mod using cv2.meanStdDev

    _, stDev = cv2.meanStdDev(mod)

    cv2.imshow('mod', mod)

    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

    if stDev > sdThresh:

        print("Motion detected.. Do something!!!")


    cv2.imshow('frame', frame2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()