import cv2 as cv
import numpy as np

def imgContains(img,pt):
    if pt[0] >= 0 and pt[0] < frame.shape[1] and pt[1] >= 0 and pt[1] < frame.shape[0]:
        return True
    else:
        return False

def euDistance(pt0,pt1):
    dif = (pt0[0]-pt1[0],pt0[1]-pt1[1])
    return np.linalg.norm(dif)

def direction(pt0,pt1):
    dif = (pt0[0]-pt1[0],pt0[1]-pt1[1])
    return np.arctan2(dif[1],dif[0])

def difAng(v0,v1):
    dif = np.abs(direction(v0[0], v0[-1]) - direction(v1[0], v1[-1]))
    if dif > np.pi:
        dif = 2*np.pi-dif
    return dif

def getCliques(grafo, features):
    n_aristas = len(grafo.getEdgeList())*2
    cliques = [[point] for point in range(len(features))]
    for i in range(0,n_aristas):
        # For every edge we get the origin and destination
        # and add them to the clique of both vertex
        origen, aux = grafo.edgeOrg(i*2)
        destino, aux = grafo.edgeDst(i*2)
        # We have to discount the first 4 vertex that make the bounding box
        origen -= 4
        destino -= 4
        if( origen >= 0 and destino >= 0):
            cliques[origen].append(destino)
            cliques[destino].append(origen)        
        
    return cliques

def getClusters(features):
    clusters = []
    #cv.flann.hierarchicalClustering(features,)

    return clusters

############# METRICS #############

def calculateMovement(features, trayectories, min_motion = 1.0):
    static_features = []
    velocity = []
    # We calculate the total length of every trayectory
    for i, (tracklet) in enumerate(trayectories):
        # #Motion of the entire trayectory
        # motion = 0
        # prev = tracklet[0]
        # for j in range(1,len(tracklet)):
        #     nex = tracklet[j]
        #     motion += euDistance(prev,nex)
        #     prev = nex

        # Distance between initial and final state
        motion = euDistance(tracklet[0],tracklet[-1])
        
        # If the length is < beta we discard it
        if motion < min_motion:
            static_features.append(i)
        # else we save its velocity
        else:
            velocity.append(motion / len(tracklet[0]))
        
    #We remove the static features
    features = np.delete(features,static_features,0)
    for i in range(len(static_features)-1,-1,-1):
        del trayectories[static_features[i]]

    return features, velocity

def calculateDirectionVar(trayectories):
    direction_variation = []
    for tracklet in trayectories:
        # Initialization
        d_var = 0
        prev_dir = None
        prev = tracklet[0]
        for j in range(1,len(tracklet)):
            # Direction
            nex = tracklet[j]
            dir = direction(prev,nex)
            prev = nex
            # Difference between past and curent direction
            if prev_dir is not None:
                d_var += np.abs(prev_dir - dir)
            prev_dir = dir
        # Average between all frames
        d_var /= len(tracklet)
        direction_variation.append(d_var)

    return direction_variation

def calculateCollectiveness(cliques, trayectories):
    # Initialization
    collectiveness = [0 for vector in trayectories]
    # For every feature point
    for k in range(len(cliques)):
        # We average the angular diference between the motion vector of every point with its neighbours
        for i in range(1,len(cliques[k])):
            collectiveness[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1]))
        collectiveness[k] /= len(cliques[k])

    return collectiveness

def calculateConflict(cliques, trayectories):
    # Initialization
    conflict = [0 for vector in trayectories]
    # For every feature point
    for k in range(len(cliques)):
        # We average quotient of the angular diference between the motion vector of every point with its neighbours and their distances
        for i in range(1,len(cliques[k])):
            conflict[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1])) / euDistance(trayectories[k][-1], trayectories[cliques[k][i]][-1])
        conflict[k] /= len(cliques[k])

    return conflict

def calculateDensity(cliques,features, bandwidth = 0.5):
    # Bandwidth = Bandwidth of the 2D Gaussian Kernel
    n_features = len(cliques)
    density = [0 for i in range(n_features)]
    for i in range(n_features):
        for j in range(1,len(cliques[i])):
            density[i] += np.exp(-1 * ( euDistance(features[i][0], features[cliques[i][j]][0]) ) / 2*bandwidth**2)
        density[i] /= np.sqrt(2*np.pi)*bandwidth
        
    return density
    
########### IMAGE VISUALIZATION ###########

def addDelaunayToImage(graph, img, color = (0,255,0), width = 1):
    triangles = graph.getTriangleList()
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if(imgContains(img,pt1) and imgContains(img,pt2) and imgContains(img,pt3)):
            cv.line(img, pt1, pt2, color, width)
            cv.line(img, pt2, pt3, color, width)
            cv.line(img, pt1, pt3, color, width)

def addTrayectoriesToImage(trayectories, img, color = (0,0,255), width = 1):
    for i, (tracklet) in enumerate(trayectories):
        prev = tracklet[0]
        prev = (int(prev[0]),int(prev[1]))
        for i in range(1,len(tracklet)):
            nex = tracklet[i]
            nex = (int(nex[0]),int(nex[1]))
            cv.line(img, prev, nex, color, width)
            prev = nex

def addCliqueToImage(cliques, index, img, features, color = (255,0,0)):
    point = features[index].ravel()
    point = (int(point[0]), int(point[1]))
    cv.circle(img,point,2,(255,0,155),4)
    for i in range(1,len(cliques[index])):
        point = features[cliques[index][i]].ravel()
        point = (int(point[0]), int(point[1]))
        cv.circle(img,point,1,color,2)
            
##################################################################

np.random.seed(1) 

# Parameters for RLOF
RLOF = cv.optflow.SparseRLOFOpticalFlow_create()
#RLOF_Param = cv.optflow.RLOFOpticalFlowParameter_create()
#RLOF.setRLOFOpticalFlowParameter(RLOF_Param)

#FAST algorithm for feature detection
fast = cv.FastFeatureDetector_create()
fast.setThreshold(25)

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("./Datasets/UMN/Crowd-Activity-All.avi")

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, prev_frame = cap.read()

# Delaunay Subdivision Function
delaunay = cv.Subdiv2D()

it = 0
L = 5  #Refresh rate
min_motion = 0.4 #Min length allowed for a trayectory

# Feature detection
prev_key = fast.detect(prev_frame,None)
prev = cv.KeyPoint_convert(prev_key)
prev = prev.reshape(-1, 1, 2)
# Trayectories initialization
trayectories = []
for p in prev:
    a, b = p.ravel()
    trayectories.append([(a,b)])

while(cap.isOpened()):
    it += 1
    
    # We calculate the metrics and begin a new set of trayectories every L frames
    if(it % L == 0):
        it = 0
        #Metrics analysis
        # Individual Behaviours
        prev, velocity = calculateMovement(prev,trayectories, min_motion)
        dir_var = calculateDirectionVar(trayectories)
                
        # Delaunay representation
        rect = (0, 0, frame.shape[1], frame.shape[0])
        delaunay.initDelaunay(rect)
        
        for point in prev:
            a, b = point.ravel()
            if(imgContains(frame,(a,b))):
                delaunay.insert((a,b))

        cliques = getCliques(delaunay, prev)

        # Interactive Behaviours
        collectiveness = calculateCollectiveness(cliques,trayectories)
        conflict = calculateConflict(cliques,trayectories)
        density = calculateDensity(cliques,prev)

        # Image representation for checking results
        addTrayectoriesToImage(trayectories,frame)
        addDelaunayToImage(delaunay,frame)
        #addCliqueToImage(cliques, -1, frame,prev)
        cv.imshow("Crowd", frame)
        
        #Beginning of a new set of trayectories
        # Feature detection
        prev_key = fast.detect(prev_frame,None)
        prev = cv.KeyPoint_convert(prev_key)
        prev = prev.reshape(-1, 1, 2)
        # Trayectories initialization
        trayectories = []
        for p in prev:
            a, b = p.ravel()
            trayectories.append([(a,b)])
    
    #ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    
    # # Calculates sparse optical flow by Lucas-Kanade method
    # # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    nex, status, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev, None)
        
    #RLOF
    #nex, status, error = RLOF.calc(prev_frame, frame, prev, None)

    if status is not None:
        
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        for i in range(len(status)-1, -1, -1):
            if status[i] == 0:
                del trayectories[i]
        
        # Selects good feature points for nex position
        good_new = nex[status == 1]
        
        for i, (new) in enumerate(good_new):
            # Adds the new coordinates to the graph and the trayectories
            a, b = new.ravel()
            if(imgContains(frame,(a,b))):
                trayectories[i].append((a,b))
    
        # Updates previous frame
        prev_frame = frame.copy()
    
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)        
    
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
