import cv2
import numpy as np


class CamshiftTracker:

    def __init__(self):
        self.hsize = 16
        self.VMin = 10
        self.VMax = 256
        self.SMin = 30
        self.firstRun = True

    def setMainImage(self, _main_image):
        self.mainImage = _main_image.copy()

    def getMainImage(self):
        return self.mainImage

    def setCurrentRect(self, curr_image, _curr_rect):
        self.currentRect = _curr_rect
        self.selection = self.currentRect

        self.mainImage = curr_image
        roi = self.mainImage[self.selection[0]:self.selection[2], self.selection[1]:self.selection[3]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        self.roiHist = cv2.normalize(self.roiHist, self.roiHist, 0, 255, cv2.NORM_MINMAX)
        self.roiBox = (self.currentRect[0], self.currentRect[1], self.currentRect[2], self.currentRect[3])



    def getCurrentRect(self):
        return self.currentRect

    def trackCurrentRect(self):
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)
        # trackBox = cv2.minAreaRect()
        # hranges = {0, 180}
        currentImage = self.mainImage
        hsv = cv2.cvtColor(currentImage, cv2.COLOR_BGR2HSV)

        # t,_,_ = cv2.split(hsv)
        # imgThreshold = np.zeros(hsv.shape, dtype=np.uint32)
        # hue = np.zeros(hsv.shape, dtype=np.uint8)
        # cv2.inRange(hsv, cv2.scaleAdd(0, self.SMin, min(self.VMin, self.VMax)),
        #             cv2.scaleAdd(180, 256, max(self.VMin, self.VMax)))
        # cv2.mixChannels([hsv], [hue], [0,0])

        # if self.firstRun:
        #     # cv::Mat roi(hue, selection), maskroi(mask, selection);
        #     roi = currentImage[self.selection[0]:self.selection[2], self.selection[1]:self.selection[3]]
        #     # maskroi = imgThreshold[self.selection[0]:self.selection[2], self.selection[1]:self.selection[3]]
        #     self.roiBox = self.selection
        #     self.roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        #     self.roiHist = cv2.normalize(self.roiHist, self.roiHist, 0, 255, cv2.NORM_MINMAX)
        #     self.firstRun = False

        # roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        # roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        self.backProj = cv2.calcBackProject([hsv], [0], self.roiHist, [0, 180], 1)
        # backProj = backProj & imgThreshold
        (r, self.roiBox) = cv2.CamShift(self.backProj, self.roiBox, termination)
        return (r, self.roiBox)







