import datetime as datetime
import numpy as np
import cv2
import time
import datetime
import pickle
import itertools
# face_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/opencv/data/haarcascades/haarcascade_eye.xml')

predict = []
measure = []
last_measure = current_measure = np.array((2,1),np.float32)
last_predict = current_predict = np.zeros((2,1),np.float32)

def kalmanSetup():
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    return kalman

def move(x,y):
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
    cv2.line(frame, (lmx,lmy), (cmx,cmy), (0,100,0))
    cv2.line(frame, (lpx,lpy), (cpx,cpy), (0,0,200))
    print(current_predict)


time_count = 0
# this is the cascade we just made. Call what you want
new_face_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/facialTracking/data/cascade.xml')

kalman = kalmanSetup()
# cap = cv2.VideoCapture('/Users/clarkpathakis/PycharmProjects/facialTracking/pos/out.mp4')
cap = cv2.VideoCapture(0)
scaling_factor = 1
start = time.time()



# forcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', forcc, 20.0, (1920, 1080))

while 1:
    ret, img = cap.read()
    frame = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # print(ret)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # add this
    # image, reject levels level weights.
    # new_faces = new_face_cascade.detectMultiScale(gray)

    face_rects = new_face_cascade.detectMultiScale(gray, 1.3, 5)

    # add this
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'X,Y is : %s, %s' % (x,y),(x,y), font, 1, (255,255,255),2)
        move(x,y)





        # line = str(datetime.datetime.now())+'\t'+str(x)+'\t'+str(y)+'\t'+str(w)+'\t'+str(h)+'\n'
        # pickle.dump(line, open("pickle_data.p", "wb"))




    # for i, p in iter(Head.items()):
    #     print("%s, %s", i, p)
    #     p.update(img)
    #     print(p.update(img))


    # out.write(img)
    cv2.imshow('img', img)
    c = cv2.waitKey(1)
    end = time.time()


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()