import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/opencv/data/haarcascades/haarcascade_eye.xml')

# this is the cascade we just made. Call what you want
new_face_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/faceTracking/data/cascade.xml')

cap = cv2.VideoCapture('/Users/clarkpathakis/PycharmProjects/faceTracking/pos/out.mp4')

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # add this
    # image, reject levels level weights.
    new_faces = new_face_cascade.detectMultiScale(gray, 50, 50)

    # add this
    for (x, y, w, h) in new_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #     # eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()