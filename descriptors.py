import pickle
import numba
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from collections import namedtuple
from numba import jit
from itertools import combinations
import visualization

############# AUXILIARY FUNCTIONS #############

@jit(nopython = True)
def imgContains(img,pt):
    return ( pt[0] > 0 and pt[0] < img.shape[1] and pt[1] > 0 and pt[1] < img.shape[0])

@jit(nopython = True)
def direction(pt0,pt1):
    dif = pt0-pt1
    if dif[0] == 0:
        return np.pi/2
    return np.arctan(dif[1]/dif[0])

@jit(nopython = True)
def difAng(v0,v1):
    dif = np.abs(direction(v0[0], v0[-1]) - direction(v1[0], v1[-1]))
    if dif >= np.pi:
        dif = 2*np.pi-dif
    return dif

@jit(nopython = True)
def delete_row(arr, num):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[num] = False
    return arr[mask]

def filter_fast_features(v_x, v_y, features, trayectories, limite):
    velocity = v_x+v_y
    
    indices = np.argsort(velocity)
    indices = indices[:-limite]
    
    features = delete_row(features,indices)
    v_x = delete_row(v_x,indices)
    v_y = delete_row(v_y,indices)
    trayectories = delete_row(trayectories,indices)

    return v_x, v_y, features , trayectories

# Calculates the area of a triangle (x2 for a better efficiency)
@jit(nopython = True)
def areaTriangle(pt0,pt1,pt2):
    return ( pt0[0] * (pt1[1] - pt2[1]) + pt1[0] * (pt2[1] - pt0[1]) + pt2[0] * (pt0[1] - pt1[0]) )

@jit( nopython = True)
def crossRatioTriangle(pt2,pt0,pt1):
    # Mid-point
    mid02 = (pt0+pt2) / 2
    mid12 = (pt1+pt2) / 2

    # Base and mid-point vectors
    e01 = pt1-pt0
    e002 = mid02-pt0
    e012 = mid12-pt0

    # Dot product
    dot02 = np.dot(e01,e002)
    dot12 = np.dot(e01,e012)

    # Distances and angles
    dst01 = np.linalg.norm(pt0-pt1)
    dst002 = np.linalg.norm(pt0-mid02)
    dst012 = np.linalg.norm(pt0-mid12)

    if (dst01 != 0 and dst002 != 0 and dst012 != 0):
        cos02 = dot02 / (dst01 * dst002)
        cos12 = dot12 / (dst01 * dst012)
        
        # Distance of the proyection to pt0
        dst_pr02 = cos02 * dst002
        dst_pr12 = cos12 * dst012

        # Cross ratio calculation
        cr = dst_pr12*(dst01 - dst_pr02) / dst01*(dst_pr12-dst_pr02)
        
    else:
        cr = 0    

    return cr

@jit( nopython = True)
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
    edges = grafo.getEdgeList()
    point = namedtuple("point", ["x", "y"])
    points = {point(features[i][0][0],features[i][0][1]):i for i in np.arange(len(features))}        
    cliques = [[] for p in features]
    for e in edges:
        # For every edge we get the origin and destination
        # and add them to the clique of both vertex
        origen = point(e[0],e[1])
        destino = point(e[2],e[3])
        if origen in points and destino in points:
            origen = points[origen]
            destino = points[destino]
            cliques[origen].append(destino)
            cliques[destino].append(origen)

    cliques = np.asarray(cliques, dtype = object)
    return cliques

def getClusters(features, mini = 2, e = 10):
    features = [item[0] for item in features]
    clusters = DBSCAN(eps=e, min_samples=mini).fit_predict(features)
    
    return clusters
 
############# METRICS #############

@jit(nopython = True)
def calculateMovement(features, trayectories, min_motion, erase_slow = False, t1 = -6):
    velocity_x = np.zeros(len(trayectories),dtype = numba.float64)
    velocity_y = np.zeros(len(trayectories),dtype = numba.float64)
    #velocity = np.zeros(len(trayectories),dtype = numba.float64)
    for i in np.arange(len(trayectories)):
        velocity_x[i] = abs(trayectories[i][t1][0]-trayectories[i][-1][0])
        velocity_y[i] = abs(trayectories[i][t1][1]-trayectories[i][-1][1])
        #velocity[i] = np.linalg.norm(trayectories[i][t1]-trayectories[i][-1]) 
        
    if erase_slow:
        static_features = np.where((velocity_x+velocity_y) < min_motion)[0]
        
    #We remove the static features
    if erase_slow:
        features = delete_row(features,static_features)
        velocity_x = delete_row(velocity_x,static_features)
        velocity_y = delete_row(velocity_y,static_features)
        trayectories = delete_row(trayectories,static_features)

    return velocity_x, velocity_y , features , trayectories

@jit(nopython = True)
def calculateDirectionVar(trayectories, t2 = 1):
    direction_variation = np.zeros(len(trayectories))

    ini = 2*t2 + (len(trayectories[0])-1) % t2
    lim = (len(trayectories[0]))

    for k in np.arange(len(trayectories)):
        for i in range(ini,lim,t2):
            direction_variation[k] += difAng((trayectories[k][i],trayectories[k][i-1]),
                                             (trayectories[k][i-1],trayectories[k][i-2]))
        direction_variation[k] /= (len(trayectories[0]) // t2)

    return direction_variation

def calculateStability(cliques, trayectories, t2 = 1):
    stability = np.zeros(len(trayectories))
    # For every tracklet
    for i in np.arange(len(trayectories)):
        # We calculate  the change of size and shape of all the posible triangles in the clique
        contador = 0
        for pair in combinations(cliques[i],2):
            if pair[0] in cliques[pair[1]]:
                contador += 1
                old_triangle = (trayectories[i][t2],trayectories[pair[0]][-1-t2], trayectories[pair[1]][-1-t2])
                new_triangle = (trayectories[i][-1],trayectories[pair[0]][-1], trayectories[pair[1]][-1])
                stability[i] += distanceTriangles(old_triangle, new_triangle)
        if contador != 0:
            stability[i] = stability[i] / contador

    return stability

@jit(nopython = True)
def auxCollectivenessAndConflict(clique,trayectories, k, t1):
    collectiveness = 0
    conflict = 0
    k = int(k)
    for i in np.arange(len(clique)):
        elem = int(clique[i])
        dif = difAng((trayectories[k][t1],trayectories[k][-1]),(trayectories[elem][t1],trayectories[elem][-1]))
        collectiveness += dif
        conflict += dif / np.linalg.norm(trayectories[k][-1] - trayectories[elem][-1])
    collectiveness /= (len(clique)+1)
    conflict /= (len(clique)+1)

    return collectiveness, conflict


def calculateCollectivenessAndConflict(cliques, trayectories,t1 = 0):
    # Initialization
    collectiveness = np.zeros(len(trayectories))
    conflict = np.zeros(len(trayectories))
    # For every feature point
    for k in np.arange(len(cliques)):
        # We measure collectiveness and conflict
        collectiveness[k], conflict[k] = auxCollectivenessAndConflict(np.array(cliques[k]),trayectories, k, t1)

    return collectiveness, conflict

@jit(nopython = True)
def auxDensity(clique, features, bandwidth, i):
    density = 0
    for j in np.arange(len(clique)):
        elem = int(clique[j])
        density += np.exp(-1 * ( np.linalg.norm(features[i][0] - features[elem][0]) ) / 2*bandwidth**2)
    density /= np.sqrt(2*np.pi)*bandwidth

    return density

def calculateDensity(cliques,features, bandwidth = 0.5):
    # Bandwidth = Bandwidth of the 2D Gaussian Kernel
    density = np.array([auxDensity(np.array(cliques[i]), features, bandwidth,i) for i in np.arange(len(cliques))])
        
    return density

@jit(nopython = True)
def auxUniformity(clique, features, f, clusters):

    inter_cluster = np.zeros(max(max(clusters)+1,1))
    intra_cluster = np.zeros(len(inter_cluster))
    total_sum = 0

    for i in np.arange(len(clique)):
        elem = int(clique[i])
        # We measure the distance if we have to
        dist = np.linalg.norm(features[f][0] - features[elem][0])
        # We follow the formula for the uniformity of each cluster
        if(f != elem):
            if(clusters[f] == clusters[elem]):
                intra_cluster[clusters[f]] += 1 / dist
            else:
                inter_cluster[clusters[f]] += 1 / dist
            total_sum += 1 / dist

    return total_sum, intra_cluster, inter_cluster

def calculateUniformity(cliques, clusters, features):
    #Initialization
    uniformity = np.zeros(max(max(clusters)+1,1))
    inter_cluster = np.zeros(len(uniformity))
    intra_cluster = np.zeros(len(uniformity))
    total_sum = 0

    # For every pair of points in each clique
    # the distance intra-cluster, inter-cluster and the total
    for f in np.arange(len(features)):
        
        total, intra, inter = auxUniformity(np.array(cliques[f]), features, f, clusters)
        total_sum += total
        intra_cluster = intra_cluster + intra
        inter_cluster = inter_cluster + inter
         

    try:
        uniformity = (intra_cluster / total_sum) - (inter_cluster / total_sum)**2
    except ZeroDivisionError:
        uniformity = np.full(max(max(clusters)+1,1),-1)


    # We return an array to keep consistency with the rest of the descriptors
    return np.array([uniformity[clusters[i]] for i in np.arange(len(clusters))])


############# MAIN FUNCTION #############

# Function to extract the descriptors of a video
def extract_descriptors(video_file, L , t1 , t2 , min_motion , fast_threshold, out_file = "descriptors", min_puntos = 10, others = {}):
    
    #Algorithm for feature detection
    if "use_sift" in others:
        # SIFT (To limit the number of features detected)
        detector = cv.SIFT_create(nfeatures = others["use_sift"])
    else:
        # FAST
        detector = cv.FastFeatureDetector_create()
        detector.setThreshold(fast_threshold)

    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(video_file)

    # How many frames will be used
    if "num_frames" in others: 
        num_frames = others["num_frames"]+L
    else:
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Skip the first frames of the video
    if "skip_frames" in others:
        for i in range(others["skip_frames"]):
            video_open, prev_frame = cap.read()
            
    video_open, prev_frame = cap.read()

    # For datasets with variable resolutions
    if "change_resolution" in others:
        height = others["change_resolution"]
        prev_frame = cv.resize(prev_frame, (int((prev_frame.shape[1]/prev_frame.shape[0]) * height), height))
    else:
        height = prev_frame.shape[0]

    width = prev_frame.shape[1]

    # Resolution related params
    clusters_e = (width+height)/50

    # Delaunay Subdivision Function
    delaunay = cv.Subdiv2D()

    data_file = open(out_file,"ab")

    it = 0

    # Feature detection
    try:
        prev_key = detector.detect(prev_frame,None)
        prev_aux = cv.KeyPoint_convert(prev_key)
        prev_aux = prev_aux.reshape(-1, 1, 2)

        # Trayectories initialization
        trayectories_aux = []
        trayectories = []
        for p in prev_aux:
            a, b = p.ravel()
            trayectories_aux.append([np.array((a,b))])
            
    except:
        pass
    
    while(video_open and it < num_frames):
        it += 1
        
        if(it % L == 0):
            # Feature detection
            try:
                prev = prev_aux.copy()
                prev_key = detector.detect(prev_frame,None)
                prev_aux = cv.KeyPoint_convert(prev_key)
                prev_aux = prev_aux.reshape(-1, 1, 2)            

                trayectories = trayectories_aux.copy()
                trayectories_aux = []
                for p in prev_aux:
                    a, b = p.ravel()
                    trayectories_aux.append([np.array((a,b))])

            except:
                trayectories = []
                trayectories_aux = []
                prev = []
        
        # We calculate the metrics and begin a new set of trayectories every L frames
        if(it >= L):
            #Metrics analysis
            if len(trayectories) > 0:
                arr_trayectories = np.array(trayectories)
                velocity_x, velocity_y, prev, arr_trayectories = calculateMovement(prev,arr_trayectories,
                                                                                   min_motion*(-t1), t1 = t1, erase_slow = True)
                # if len(prev) > max_num_features and max_num_features != -1:
                #     print("Limite sobrepasado {}".format(len(prev)))
                #     velocity_x, velocity_y, prev, arr_trayectories = filter_fast_features(velocity_x, velocity_y,
                #                                                                           prev, arr_trayectories, max_num_features)
                trayectories = arr_trayectories.tolist()                

            if len(prev) > min_puntos:
                dir_var = calculateDirectionVar(arr_trayectories, t2 = t2)                
                # Delaunay representation
                rect = (0, 0, prev_frame.shape[1], prev_frame.shape[0])
                delaunay.initDelaunay(rect)
                for point in prev:
                    a, b = point.ravel()[:2]
                    if(imgContains(frame,(a,b))):
                        delaunay.insert((a,b))
                
                cliques = getCliques(delaunay, prev)
                
                # Interactive Behaviours
                stability = calculateStability(cliques,arr_trayectories)
                collectiveness, conflict = calculateCollectivenessAndConflict(cliques,arr_trayectories, t1 = t1)
                density = calculateDensity(cliques,prev)
                
                clusters = getClusters(prev, e = clusters_e)

                uniformity = calculateUniformity(cliques, clusters, prev)
                    
                # Image representation for visualizing results
                #visualization.addTrayectoriesToImage(trayectories,frame)
                visualization.addDelaunayToImage(delaunay,frame)
                visualization.addCliqueToImage(cliques, -1, frame,trayectories)
                #visualization.addClustersToImage(clusters,prev,frame)

            else:
                velocity_x = np.zeros(1)
                velocity_y = np.zeros(1)
                dir_var = np.zeros(1)
                collectiveness = np.zeros(1)
                conflict = np.zeros(1)
                stability = np.zeros(1)
                density = np.zeros(1)
                uniformity = np.zeros(1)

            # We save the data into a file
            descriptores = [velocity_x, velocity_y, dir_var, stability, collectiveness, conflict, density, uniformity]
            pickle.dump(descriptores, data_file)

            cv.imshow("Crowd", frame)            

        video_open, frame = cap.read()
        
        if video_open:
            if frame.shape[0] != height:
                frame = cv.resize(frame, (int((frame.shape[1]/frame.shape[0]) * height), height))
            
            # Sparse optical flow by Lucas-Kanade method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        
            if (it > L):
                try:
                    nex, status, _ = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev, None)
                    _, status, _ = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex, prev)
                    
                    # Selects good feature points for previous position
                    for i in np.arange(len(status)-1, -1, -1):
                        if status[i] == 0:
                            del trayectories[i]
                
                    # Selects good feature points for nex position
                    good_new = nex[status == 1]
        
                    for i, (new) in enumerate(good_new):
                        # Adds the new coordinates to the graph and the trayectories
                        a, b = new.ravel()
                        trayectories[i].append(np.array((a,b)))
                        del trayectories[i][0]
    
                    # Updates previous good feature points
                    prev = good_new.reshape(-1, 1, 2)

                except:
                    pass

            try:
                nex_aux, status_aux, _ = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev_aux, None)
                _, status_aux, _ = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex_aux, prev_aux)
        
                # Selects good feature points for previous position
                for i in np.arange(len(status_aux)-1, -1, -1):
                    if status_aux[i] == 0:
                        del trayectories_aux[i]
                
                # Selects good feature points for nex position
                good_new = nex_aux[status_aux == 1]
        
                for i, (new) in enumerate(good_new):
                    # Adds the new coordinates to the graph and the trayectories
                    trayectories_aux[i].append(np.array((new[0],new[1])))
                    if( len(trayectories_aux[i]) > L ):
                        del trayectories_aux[i][0]
        
                # Updates previous good feature points
                prev_aux = good_new.reshape(-1, 1, 2)

                # Updates previous frame
                prev_frame = frame.copy()
            except:
                pass
    
        # Frames are read by intervals of 1 milliseconds. The function ends after the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
    data_file.close()

