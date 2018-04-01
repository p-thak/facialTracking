# import the necessary packages
import numpy as np
import argparse
import cv2

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False

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

kalman = kalmanSetup()

def applyKalmanFilter(x, y):
    global frame,last_measure,current_measure,measure,current_predict,last_predict,kalman
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
    # print(current_measure)
    # print(current_predict)



def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
    args = vars(ap.parse_args())

    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if the video path was not supplied, grab the reference to the
    # camera

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
        # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
        # setup the mouse callback

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)
    kalman = kalmanSetup()

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)
    roiBox = None

    forcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output1i.avi', forcc, 20.0, (1920, 1080))

    # keep looping over the frames
    while True:

        # grab the current frame
        (grabbed, frame) = camera.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # if the see if the ROI has been computed
        if roiBox is not None:
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.cv2.boxPoints(r))
            applyKalmanFilter(roiBox[0],roiBox[1])
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # show the frame and record if the user presses a key
        # out.write(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # handle if the 'i' key is pressed, then go into ROI
        # selection mode
        if key == ord("i") and len(roiPts) < 4:
            # indicate that we are in input mode and clone the
            # frame
            inputMode = True
            orig = frame.copy()

            # keep looping until 4 reference ROI points have
            # been selected; press any key to exit ROI selction
            # mode once 4 points have been selected
            while len(roiPts) < 4:
                 cv2.imshow("frame", frame)
                 cv2.waitKey(0)

            # determine the top-left and bottom-right points
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis = 1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]

            # grab the ROI for the bounding box and convert it
            # to the HSV color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

            # compute a HSV histogram for the ROI and store the
            # bounding box
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])

        # if the 'q' key is pressed, stop the loop
        elif key == ord("q"):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()