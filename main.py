import cv2 as cv
import numpy as np

def imgContains(img,pt):
    if pt[0] >= 0 and pt[0] < frame.shape[1] and pt[1] >= 0 and pt[1] < frame.shape[0]:
        return True
    else:
        return False

def printDelaunay(graph, img, message = "Delaunay Triangulation", color = (0,255,0)):
    triangles = graph.getTriangleList()
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if(imgContains(img,pt1) and imgContains(img,pt2) and imgContains(img,pt3)):
            cv.line(img, pt1, pt2, color, 1)
            cv.line(img, pt2, pt3, color, 1)
            cv.line(img, pt1, pt3, color, 1)
    cv.imshow(message, img)

##################################################################

# Parameters for RLOF
RLOF = cv.optflow.SparseRLOFOpticalFlow_create()

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("./Datasets/UMN/Crowd-Activity-All.avi")

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, prev_frame = cap.read()

# Delaunay Subdivision Function
delaunay = cv.Subdiv2D()

#FAST algorithm for feature detection
print("Detectando Features...")
fast = cv.FastFeatureDetector_create()
fast.setThreshold(50)
prev_key = fast.detect(prev_frame,None)
prev = cv.KeyPoint_convert(prev_key)

print("Comenzando Tracking:")
while(cap.isOpened()):
    #ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    
    # # Calculates sparse optical flow by Lucas-Kanade method
    # # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    nex, status, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev, None)
    
    #RLOF
    #nex, status, error = RLOF.calc(prev_frame, frame, prev, None)

    # Selects good feature points for previous position
    if(len(prev.shape) == 3):
        good_old = prev[status == 1]
    else:
        # All features are good at the beginning
        good_old = prev
        
    # Selects good feature points for nex position
    if(len(nex.shape) == 3):
        good_new = nex[status == 1]
    else:
        good_new = nex
        
    # Creation of the Delaunay graph
    rect = (0, 0, prev_frame.shape[1], frame.shape[0])
    delaunay.initDelaunay(rect)
    for i, (new) in enumerate(good_new):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        #print(new)
        if(imgContains(frame,(a,b))):
            delaunay.insert((a,b))
    
    # Updates previous frame
    prev_frame = frame.copy()
    
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    
    # Opens a new window and displays the output frame
    printDelaunay(delaunay,frame)
    
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
