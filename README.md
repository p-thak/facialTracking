# facialTracking
Uses facial tracking to track a person's face even when they turn around. 

To run: python3 HaarCascadeVideo.py --shape-predictor shape_predictor_68_face_landmarks.dat --video "path/to/video"

** Note: to run facial tracking from your camera, do not include the last parameter.

HaarCascadeVideo.py runs first a facial recognition and then facial tracking withing the program. The facial recognition piece uses a classifier that we trained using Haar Cascade and OpenCV. From there, it then takes a portion of the bounding box created from the facial recogntion and sets that as the region of interest. We use that region of interest to run CAMshift and Kalman filters. The CAMshift filter is a variation on the Meanshift filter which uses pixel distribution for histogram backprojection, but then adjusts the size of the bounding box that tracks the face. The Kalman filter then smooths out the bouding box to make it less jittery. 





