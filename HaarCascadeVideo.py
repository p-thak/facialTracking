import datetime as datetime
import numpy as np
import cv2
import time
import os
import datetime
import pickle
import itertools

predict = []
measure = []
last_measure = current_measure = np.array((2,1),np.float32)
last_predict = current_predict = np.zeros((2,1),np.float32)
frame = np.ndarray
# measure = np.array((2,1),np.float32)

def kalmanSetup():
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    return kalman

def applyCamshiftFilter(x,y,w,h,termination):
    roi = frame[y:y+h,x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    roiHist = cv2.calcHist([roi],[0],None,[16],[0,180])
    roiHist = cv2.normalize(roiHist,roiHist, 0 ,255, cv2.NORM_MINMAX)
    roiBox = (x,y,x+w,y+h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0], roiHist, [0,180], 1)

    (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
    pts = np.int0(cv2.cv2.boxPoints(r))
    cv2.polylines(frame, [pts], True, (0,255,0), 2)



# applies the Kalman filter

def applyKalmanFilter(x, y):
    global frame,last_measure,current_measure,measure,current_predict,last_predict
    last_predict=current_predict
    last_measure=current_measure
    predict.append([int(last_predict[0]),int(last_predict[1])])
    measure.append([int(last_predict[0]),int(last_predict[1])])
    current_measure=np.array([[np.float32(x)],[np.float32(y)]])
    kalman.correct(current_measure)
    current_predict = kalman.predict()
    lmx,lmy=last_measure[0],last_measure[1]
    cmx,cmy=current_measure[0],current_measure[1]
    cpx,cpy=current_predict[0],current_predict[1]
    lpx,lpy=last_predict[0],last_predict[1]

    cv2.line(frame, (lmx,lmy), (cmx,cmy), (0,100,0), 2)
    cv2.line(frame, (lpx,lpy), (cpx,cpy), (0,0,200), 2)



time_count = 0
# this is the cascade we just made. Call what you want
dir_path = os.path.dirname(os.path.realpath(__file__))
new_face_cascade = cv2.CascadeClassifier(dir_path+'/smallData/cascade.xml')

kalman = kalmanSetup()

# cap = cv2.VideoCapture('/Users/clarkpathakis/PycharmProjects/facialTracking/pos/out.mp4')
cap = cv2.VideoCapture(0)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)
start = time.time()
roiBox = None
# forcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while 1:
    ret, img = cap.read()
    # frame = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    frame = img

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # image, reject levels level weights.
    # The scale factor and minNeighbors need to be adjusted according to lighting and background.
    face_rects = new_face_cascade.detectMultiScale(gray, 1.9, 9)
    if roiBox is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv],[0], roiHist, [0,180], 1)

        (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
        pts = np.int0(cv2.cv2.boxPoints(r))
        applyKalmanFilter(roiBox[0], roiBox[1])
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    else:
        for (x, y, w, h) in face_rects:
            # takes a smaller portion out of the facial recognition bounding box and then has those points
            # for the region of interest, ready to apply kalman filter and CAMshift filter with next frame
            remainder = int(w * .35)
            remainder_y = int(h * .35)
            new_w = int(w * .65)
            new_h = int(h * .65)
            new_x = int(x + (remainder / 2))
            new_y = int(y + (remainder_y / 2))
            top_left = (new_x,new_y)
            bottom_right = (new_x+new_w, new_y+new_h)

            orig = frame.copy()
            roi = orig[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # compute a HSV histogram for the ROI and store the
            # bounding boxi
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (x, y, bottom_right[0], bottom_right[1])
            # roiBox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])


    # This shows a bounding box just as a reference for where the facial recognition is.

    # for (x, y, w, h) in face_rects:
    #     # applyCamshiftFilter(x,y,w,h,termination)
    #     # applyKalmanFilter(x, y)
    #     # 85%
    #     remainder = int(w*.35)
    #     remainder_y = int(h * .35)
    #     new_w = int(w*.65)
    #     new_h = int(h * .65)
    #     new_x = int(x+(remainder/2))
    #     new_y = int(y + (remainder_y / 2))
    #     cv2.line(img, (x,y),(new_x,new_y+new_h), (0,0,255), 3)
    #     cv2.line(img, (x+w,y),(x,y), (0,0,255), 3)
    #     cv2.line(img, (x+w,y), (new_x+new_w, new_y+new_h), (0,0,255), 3)
    #     cv2.line(img, (new_x, new_y+new_h), (new_x+new_w, new_y+new_h), (0,0,255), 3)
    #     # cv2.rectangle(img, (new_x, y), (new_x+new_w, new_y+new_h), (255, 0, 0), 3)
    #     # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img,'X,Y is : %s, %s' % (x,y),(x,y), font, 1, (255,255,0),2)




    cv2.imshow('img', img)
    out.write(img)
    c = cv2.waitKey(1)
    end = time.time()


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()