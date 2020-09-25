import cv2 as cv
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import time

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

def areaTriangle(pt0,pt1,pt2):
    #Area of the triangle (x2 for a better efficiency)
    area = pt0[0] * (pt1[1] - pt2[1]) + pt1[0] * (pt2[1] - pt0[1]) + pt2[0] * (pt0[1] - pt1[0])
    
    return area

def crossRatioTriangle(pt2,pt0,pt1):
    # Mid-point
    mid02 = ((pt2[0]+pt0[0]) / 2, (pt2[1]+pt0[1]) / 2)
    mid12 = ((pt2[0]+pt1[0]) / 2, (pt2[1]+pt1[1]) / 2)

    # Base and mid-point vectors
    e01 = (pt1[0] - pt0[0], pt1[1] - pt0[1])
    e002 = (mid02[0] - pt0[0], mid02[1] - pt0[1])
    e012 = (mid12[0] - pt0[0], mid12[1] - pt0[1])

    # Dot product
    dot02 = np.dot(e01,e002)
    dot12 = np.dot(e01,e012)

    # Distances and angles
    dst01 = euDistance(pt0, pt1)
    dst002 = euDistance(pt0, mid02)
    dst012 = euDistance(pt0, mid12)
    if (dst01 * dst002 * dst012 > 0):
        cos02 = dot02 / (dst01 * dst002)
        cos12 = dot12 / (dst01 * dst012)
    else:
        return 0

    # Distance of the proyection to pt0
    dst_pr02 = cos02 * dst002
    dst_pr12 = cos12 * dst012

    # Cross ratio calculation
    cr = dst_pr12*(dst01 - dst_pr02) / dst01*(dst_pr12-dst_pr02)

    return cr

def distanceTriangles(old, new):
    # Area
    old_area = areaTriangle(old[0], old[1], old[2])
    new_area = areaTriangle(new[0], new[1], new[2])
    diff_area = old_area - new_area

    # Cross ratio
    if(old_area > 0 and new_area > 0):
        diff_cr = crossRatioTriangle(old[0], old[1], old[2]) - crossRatioTriangle(new[0], new[1], new[2])
    else:
        diff_cr = 1

    return np.abs(diff_area)*np.abs(diff_cr)

def getCliques(grafo, features):
    n_aristas = len(grafo.getEdgeList())
    cliques = [[point] for point in range(len(features))]
    limit = n_aristas
    i = 0
    while i < limit:
        # For every edge we get the origin and destination
        # and add them to the clique of both vertex
        origen, aux = grafo.edgeOrg(i*4) # Odd positions have origin and destination 0
        destino, aux = grafo.edgeDst(i*4) # Also, edges are duplicated (one for every orientation)
        # We have to discount the first 4 vertex that make the bounding box
        origen -= 4
        destino -= 4
        if( origen >= 0 and destino >= 0):
            cliques[origen].append(destino)
            cliques[destino].append(origen)
        i += 1
        
    return cliques

def getClusters(features):
    features = [item[0] for item in features]
    #clusters = AffinityPropagation(random_state = 1, copy = False).fit_predict(features)
    clusters = DBSCAN(eps=10, min_samples=3).fit_predict(features)
    
    return clusters

# Function to add new features to an existing array
# Can be used for a more stable graph but execution time increases 
def addFeatures(prev_features, new_features, accuracy = 5.0):
    add = [1 for f in new_features]
    for k in range(len(new_features)):
        for i in range(len(prev_features)):
            if np.abs(new_features[k][0] - prev_features[i][0][0]) < accuracy and np.abs(new_features[k][1] - prev_features[i][0][1]) < accuracy:
                add[k] = 0
                break

    new_features = new_features[np.nonzero(add)]
    if(len(new_features) == 0):
        return prev_features
    new_features = new_features.reshape(-1, 1, 2)
            
    return np.append(prev_features,new_features, axis = 0)

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

def calculateStability(cliques, trayectories):
    stability = [0 for f in trayectories]
    # For every tracklet
    for i in range(len(trayectories)):
        # We calculate  the change of size and shape of all the posible triangles in the clique 
        for j in range(1,len(cliques[i])):
            for k in range(j+1,len(cliques[i])):
                old_triangle = (trayectories[i][0],trayectories[cliques[i][j]][0], trayectories[cliques[i][k]][0])
                new_triangle = (trayectories[i][-1],trayectories[cliques[i][j]][-1], trayectories[cliques[i][k]][-1])
                stability[i] += distanceTriangles(old_triangle, new_triangle)
        stability[i] /= len(cliques[i])

    return stability
            

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

def calculateUniformity(cliques, clusters, features):
    #Initialization 
    uniformity = [0 for i in range(max(clusters)+1)]
    inter_cluster = [0 for i in uniformity]
    intra_cluster = inter_cluster.copy()
    total_sum = 0
    dist_matrix = [[-1 for f2 in features] for f1 in features]

    # For every pair of point in each clique
    for f in range(len(features)):
        for q in range(1,len(cliques[f])):
            # We measure the distance if we have to
            if dist_matrix[f][cliques[f][q]] == -1:
                    dist_matrix[f][cliques[f][q]] = euDistance(features[f][0], features[cliques[f][q]][0])
                    dist_matrix[cliques[f][q]][f] = dist_matrix[f][cliques[f][q]]
            # We follow the formula for the uniformity of each cluster
            if(clusters[f] == clusters[cliques[f][q]]):
                intra_cluster[clusters[f]] += 1 / dist_matrix[f][cliques[f][q]]
            else:
                inter_cluster[clusters[f]] += 1 / dist_matrix[f][cliques[f][q]]
            total_sum += 1 / dist_matrix[f][cliques[f][q]]

    for i in range(len(uniformity)):
        uniformity[i] = (intra_cluster / total_sum) - (inter_cluster / total_sum)**2
            
    return uniformity
    
########### IMAGE VISUALIZATION ###########

def addDelaunayToImage(graph, img, color = (0,255,0), width = 1):
    triangles = graph.getEdgeList()
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        if(imgContains(img,pt1) and imgContains(img,pt2)):
            cv.line(img, pt1, pt2, color, width)

def addTrayectoriesToImage(trayectories, img, color = (0,0,255), width = 1):
    for i, (tracklet) in enumerate(trayectories):
        prev = tracklet[0]
        prev = (int(prev[0]),int(prev[1]))
        for i in range(1,len(tracklet)):
            nex = tracklet[i]
            nex = (int(nex[0]),int(nex[1]))
            cv.line(img, prev, nex, color, width)
            prev = nex

def addCliqueToImage(cliques, index, img, trayectories, tr_index = -1, color = (255,0,0)):
    point = trayectories[index][tr_index]
    point = (int(point[0]), int(point[1]))
    cv.circle(img,point,2,(255,0,155),4)
    for i in range(1,len(cliques[index])):
        point = trayectories[cliques[index][i]][tr_index]
        point = (int(point[0]), int(point[1]))
        cv.circle(img,point,1,color,2)

def addClustersToImage(clusters, features, img):
    n_clusters = max(clusters)+1
    for i in range(len(clusters)):
        point = features[i].ravel()
        point = (int(point[0]), int(point[1]))
        color = 255 * (clusters[i]+1) / n_clusters
        cv.circle(img,point,1,(color,0,color),2)
            
##################################################################

#np.random.seed(1) 

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
min_motion = 0.2 #Min length allowed for a trayectory

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
    if(it >= L):
        it = 0
        
        #Metrics analysis
        # Individual Behaviours
        start = time.time()
        prev, velocity = calculateMovement(prev,trayectories, min_motion)
        dir_var = calculateDirectionVar(trayectories)

        if len(prev) > 0:
            # Delaunay representation
            rect = (0, 0, frame.shape[1], frame.shape[0])
            delaunay.initDelaunay(rect)
        
            for point in prev:
                a, b = point.ravel()
                if(imgContains(frame,(a,b))):
                    delaunay.insert((a,b))

            cliques = getCliques(delaunay, prev)

            # Interactive Behaviours
            stability = calculateStability(cliques,trayectories)
            collectiveness = calculateCollectiveness(cliques,trayectories)
            conflict = calculateConflict(cliques,trayectories)
            density = calculateDensity(cliques,prev)
            clusters = getClusters(prev)
            uniformity = calculateUniformity(cliques, clusters, prev)
            print(time.time()-start)

            # Image representation for checking results
            #addTrayectoriesToImage(trayectories,frame)
            addDelaunayToImage(delaunay,frame)
            #addCliqueToImage(cliques, 1, frame,trayectories)
            #addClustersToImage(clusters,prev,frame)
            cv.imshow("Crowd", frame)
        
        #Beginning of a new set of trayectories
        # Feature detection
        # new_key = fast.detect(prev_frame,None)
        # new = cv.KeyPoint_convert(new_key)
        # prev = addFeatures(prev,new,10.0)

        prev_key = fast.detect(prev_frame,None)
        prev = cv.KeyPoint_convert(prev_key)
        prev = prev.reshape(-1,1,2)        
        
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
