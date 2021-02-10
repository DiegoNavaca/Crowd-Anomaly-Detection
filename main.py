import time
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from collections import namedtuple
from glob import glob
import joblib
import numba
from numba import jit
from itertools import combinations

# Reads the ground truth from a file
def get_Ground_Truth(in_file):
    f = open(in_file)
    gt = {}
    for line in f:
        lista = line.split(",")
        # name: beginning anomaly
        gt[lista[0]] = [int(lista[1]),int(lista[2])]

    f.close()
        
    return gt

@jit(nopython = True)
def imgContains(img,pt):
    return ( pt[0] >= 0 and pt[0] < img.shape[1] and pt[1] >= 0 and pt[1] < img.shape[0] )

@jit(nopython = True)
def direction(pt0,pt1):
    dif = pt0-pt1
    return np.arctan2(dif[1],dif[0])

@jit(nopython = True)
def difAng(v0,v1):
    dif = np.abs(direction(v0[0], v0[-1]) - direction(v1[0], v1[-1]))
    if dif > np.pi:
        dif = 2*np.pi-dif
    return dif

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

# Function to add new features to an existing array
# Can be used for a more stable graph but execution time increases 
def addFeatures(prev_features, new_features, accuracy = 5.0):
    grid = prev_features // accuracy
    point = namedtuple("point", ["x", "y"])
    points = frozenset(point(grid[i][0][0],grid[i][0][1]) for i in np.arange(len(grid)))
    add = [1 for f in new_features]
    for k in np.arange(len(new_features)):
        p = point(new_features[k][0] // accuracy, new_features[k][1] // accuracy)
        if p in points:
            add[k] = 0

    new_features = new_features[np.nonzero(add)]
    if(len(new_features) == 0):
        return prev_features
    new_features = new_features.reshape(-1, 1, 2)
            
    return np.append(prev_features,new_features, axis = 0)

############# METRICS #############
@jit(nopython = True)
def delete_row(arr, num):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[num] = False
    return arr[mask]

@jit(nopython = True)
def calculateMovement(features, trayectories, persp_map, min_motion = 1.0, erase_slow = False, t1 = -6):
    #velocity = np.array([np.linalg.norm(tracklet[0]-tracklet[-1]) for tracklet in trayectories])
    velocity = np.zeros(len(trayectories),dtype = numba.float64)
    for i in range(len(trayectories)):
        velocity[i] = np.linalg.norm(np.dot(persp_map,trayectories[i][t1])-np.dot(persp_map,trayectories[i][-1]) )
        
    if erase_slow:
        static_features = np.where(velocity < min_motion)[0]
        
    #We remove the static features
    if erase_slow:
        features = delete_row(features,static_features)
        velocity = delete_row(velocity,static_features)
        trayectories = delete_row(trayectories,static_features)

    velocity = velocity / np.array([len(tracklet) for tracklet in trayectories])

    return features, velocity, trayectories

@jit(nopython = True)
def calculateDirectionVar(trayectories, t2 = 1):
    direction_variation = np.zeros(len(trayectories))

    ini = 2*t2 + (len(trayectories[0])-1) % t2
    lim = (len(trayectories[0]) // t2)-2

    for k in range(len(trayectories)):
        for i in range(ini,lim,t2):
            aux = np.abs( direction(trayectories[k][i*t2],trayectories[k][(i-1)*t2])
                                              - direction(trayectories[k][(i-1)*t2],trayectories[k][(i-2)*t2]) )
            if aux >= np.pi:
                aux = 2*np.pi-aux
                
            direction_variation[k] += aux
        direction_variation[k] /= (len(trayectories[0]) // t2)

    return direction_variation

def calculateStability(cliques, trayectories, t2 = 1):
    stability = np.zeros(len(trayectories))
    # For every tracklet
    for i in np.arange(len(trayectories)):
        # We calculate  the change of size and shape of all the posible triangles in the clique 
        for pair in combinations(cliques[i],2):
            old_triangle = (trayectories[i][t2],trayectories[pair[0]][-1-t2], trayectories[pair[1]][-1-t2])
            new_triangle = (trayectories[i][-1],trayectories[pair[0]][-1], trayectories[pair[1]][-1])
            stability[i] += distanceTriangles(old_triangle, new_triangle)
        stability[i] = stability[i] / (len(cliques[i])+1)

    return stability

# def calculateCollectiveness(cliques, trayectories):
#     # Initialization
#     collectiveness = [0 for vector in trayectories]
#     # For every feature point
#     for k in np.arange(len(cliques)):
#         # We average the angular diference between the motion vector of every point with its neighbours
#         for elem in cliques[k]:
#             collectiveness[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[elem][0],trayectories[elem][-1]))
#         collectiveness[k] /= (len(cliques[k])+1)

#     return collectiveness

# def calculateConflict(cliques, trayectories):
#     # Initialization
#     conflict = [0 for vector in trayectories]
#     # For every feature point
#     for k in np.arange(len(cliques)):
#         # We average quotient of the angular diference between the motion vector of every point with its neighbours and their distances
#         for elem in cliques[k]:
#             conflict[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[elem][0],trayectories[elem][-1])) / np.linalg.norm(trayectories[k][-1] - trayectories[elem][-1])
#         conflict[k] /= (len(cliques[k])+1)

#     return conflict

@jit(nopython = True)
def auxCollectivenessAndConflict(clique,trayectories, k, t1):
    collectiveness = 0
    conflict = 0
    k = int(k)
    for i in range(len(clique)):
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
    conflict = collectiveness.copy()
    # For every feature point
    for k in range(len(cliques)):
        # We measure collectiveness and conflict
        collectiveness[k], conflict[k] = auxCollectivenessAndConflict(np.array(cliques[k]),trayectories, k, t1)

    return collectiveness, conflict

@jit(nopython = True)
def auxDensity(clique, features, bandwidth, i):
    density = 0
    for j in range(len(clique)):
        elem = int(clique[j])
        density += np.exp(-1 * ( np.linalg.norm(features[i][0] - features[elem][0]) ) / 2*bandwidth**2)
    density /= np.sqrt(2*np.pi)*bandwidth

    return density

def calculateDensity(cliques,features, bandwidth = 0.5):
    # Bandwidth = Bandwidth of the 2D Gaussian Kernel
    density = np.array([auxDensity(np.array(cliques[i]), features, bandwidth,i) for i in range(len(cliques))])
    
    return density

@jit(nopython = True)
def auxUniformity(clique, features, f, clusters):
    dist_matrix = np.zeros((len(features),len(features)),dtype = numba.float64)

    inter_cluster = np.zeros(max(max(clusters)+1,1))
    intra_cluster = inter_cluster.copy()
    total_sum = 0

    for i in np.arange(len(clique)):
        elem = int(clique[i])
        # We measure the distance if we have to
        if dist_matrix[f][elem] == 0:
            dist_matrix[f][elem] = np.linalg.norm(features[f][0] - features[elem][0])
            dist_matrix[elem][f] = dist_matrix[f][elem]
        # We follow the formula for the uniformity of each cluster
        if(f != elem):
            if(clusters[f] == clusters[elem]):
                intra_cluster[clusters[f]] += 1 / dist_matrix[f][elem]
            else:
                inter_cluster[clusters[f]] += 1 / dist_matrix[f][elem]
            total_sum += 1 / dist_matrix[f][elem]

    return total_sum, intra_cluster, inter_cluster

def calculateUniformity(cliques, clusters, features):
    #Initialization 
    uniformity = np.zeros(max(max(clusters)+1,1))
    inter_cluster = np.zeros(len(uniformity))
    intra_cluster = inter_cluster.copy()
    total_sum = 0
    

    # For every pair of point in each clique
    for f in np.arange(len(features)):
        total, intra, inter = auxUniformity(np.array(cliques[f]), features, f, clusters)
        total_sum += total
        intra_cluster = intra_cluster + intra
        inter_cluster = inter_cluster + inter

    if(total_sum == 0):
        uniformity = np.full(max(max(clusters)+1,1),-1)
    else:
        uniformity = (intra_cluster / total_sum) - (inter_cluster / total_sum)**2

    # We return a list to keep consistency with the rest of the descriptors
    return np.array([uniformity[clusters[i]] for i in np.arange(len(clusters))])
    #return uniformity
    
########### IMAGE VISUALIZATION ###########

def addDelaunayToImage(graph, img, color = (0,255,0), width = 1):
    triangles = graph.getEdgeList()
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        if(imgContains(img,pt1) and imgContains(img,pt2)):
            cv.line(img, pt1, pt2, color, width)

def addTrayectoriesToImage(trayectories, img, color = (0,0,255), width = 1):
    for tracklet in trayectories:
        prev = tracklet[0]
        prev = (int(prev[0]),int(prev[1]))
        for j in np.arange(1,len(tracklet)):
            nex = tracklet[j]
            nex = (int(nex[0]),int(nex[1]))
            cv.line(img, prev, nex, color, width)
            prev = nex

def addCliqueToImage(cliques, index, img, trayectories, tr_index = -1, color = (255,0,0)):
    point = trayectories[index][tr_index]
    point = (int(point[0]), int(point[1]))
    cv.circle(img,point,2,(255,0,155),4)
    for i in np.arange(len(cliques[index])):
        point = trayectories[cliques[index][i]][tr_index]
        point = (int(point[0]), int(point[1]))
        cv.circle(img,point,1,color,2)

def addClustersToImage(clusters, features, img):
    n_clusters = max(clusters)+1
    for i in np.arange(len(clusters)):
        point = features[i].ravel()
        point = (int(point[0]), int(point[1]))
        color = 255 * (clusters[i]+1) / n_clusters
        cv.circle(img,point,1,(color,0,color),2)
            
#################### DESCRIPTORS ###########################

# Function to extract the descriptors of a video
def extract_descriptors(video_file, persp_map, out_file = "descriptors"):
    #FAST algorithm for feature detection
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(fast_threshold)
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(video_file)

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    video_open, prev_frame = cap.read()
    #prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY);

    # Delaunay Subdivision Function
    delaunay = cv.Subdiv2D()

    file = open(out_file,"a")

    it = 0

    # Feature detection
    prev_key = fast.detect(prev_frame,None)
    prev_aux = cv.KeyPoint_convert(prev_key)
    prev_aux = prev_aux.reshape(-1, 1, 2)

    # Trayectories initialization
    trayectories_aux = []
    trayectories = []
    for p in prev_aux:
        a, b = p.ravel()
        trayectories_aux.append([np.array((a,b,1))])
    
    while(video_open):
        it += 1
        
        if(it % L == 0):
            # Feature detection
            prev = prev_aux.copy()
            prev_key = fast.detect(prev_frame,None)
            prev_aux = cv.KeyPoint_convert(prev_key)
            prev_aux = prev_aux.reshape(-1, 1, 2)

            trayectories_aux = np.array(trayectories_aux)
            prev, velocity, trayectories = calculateMovement(prev,trayectories_aux, persp_map, min_motion*L, erase_slow = True, t1 =  t1)
            trayectories = trayectories.tolist()
            trayectories_aux = []
            for p in prev_aux:
                a, b = p.ravel()
                trayectories_aux.append([np.array((a,b,1))])
        
        # We calculate the metrics and begin a new set of trayectories every L frames
        if(it >= L):
            #Metrics analysis
            if len(trayectories) > 0:
                arr_trayectories = np.array(trayectories)
                _, velocity, _ = calculateMovement(prev,arr_trayectories, persp_map, min_motion*L, t1 = t1)
                

            if len(prev) > 10:

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

                clusters = getClusters(prev)
                uniformity = calculateUniformity(cliques, clusters, prev)

            else:
                #print("None detected")
                velocity = np.zeros(1)
                dir_var = velocity.copy()
                collectiveness = velocity.copy()
                conflict = velocity.copy()
                stability = velocity.copy()
                density = velocity.copy()
                uniformity = velocity.copy()
                
                                
                # Buscar una forma mejor
                
            file.write(str(velocity.tolist())+"\n")
            file.write(str(dir_var.tolist())+"\n")
            file.write(str(stability.tolist())+"\n")
            file.write(str(collectiveness.tolist())+"\n")
            file.write(str(conflict.tolist())+"\n")
            file.write(str(density.tolist())+"\n")
            file.write(str(uniformity.tolist())+"\n")                

            # Image representation for checking results
            addTrayectoriesToImage(trayectories,frame)
            #addDelaunayToImage(delaunay,frame)
            #addCliqueToImage(cliques, -1, frame,trayectories)
            #addClustersToImage(clusters,prev,frame)
            #if frame.shape[0] < 512:
            #    frame = cv.resize(frame,(512,int(512*frame.shape[0]/frame.shape[1])))
            cv.imshow("Crowd", frame)            
        
        #ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        video_open, frame = cap.read()
        
        if video_open:
            #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY);
            # # Calculates sparse optical flow by Lucas-Kanade method
            # # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
            if (it > L) :
                if len(prev) > 0:
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
                        trayectories[i].append(np.array((a,b,1)))
                        del trayectories[i][0]
    
                    # Updates previous good feature points
                    prev = good_new.reshape(-1, 1, 2)

            nex_aux, status_aux, _ = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev_aux, None)
            _, status_aux, _ = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex_aux, prev_aux)

            if status_aux is not None:
        
                # Selects good feature points for previous position
                for i in np.arange(len(status_aux)-1, -1, -1):
                    if status_aux[i] == 0:
                        del trayectories_aux[i]
                
                # Selects good feature points for nex position
                good_new = nex_aux[status_aux == 1]
        
                for i, (new) in enumerate(good_new):
                    # Adds the new coordinates to the graph and the trayectories
                    a, b = new.ravel()
                    trayectories_aux[i].append(np.array((a,b,1)))
                    if( len(trayectories_aux[i]) > L ):
                        del trayectories_aux[i][0]
        
                # Updates previous good feature points
                prev_aux = good_new.reshape(-1, 1, 2)

                # Updates previous frame
                prev_frame = frame.copy()
    
    # Frames are read by intervals of 1 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
    file.close()

# Function to get the descriptors of a set of training videos and store them on a single file
def get_Training_Descriptors(path,persp_map, out_file = "training_descriptors"):
    # We reset the file if it exists
    f = open(out_file,"w")
    
    velocity = np.zeros(1)
    dir_var = velocity.copy()
    collectiveness = velocity.copy()
    conflict = velocity.copy()
    stability = velocity.copy()
    density = velocity.copy()
    uniformity = velocity.copy()
                
                                
    # Buscar una forma mejor
                
    f.write(str(velocity.tolist())+"\n")
    f.write(str(dir_var.tolist())+"\n")
    f.write(str(stability.tolist())+"\n")
    f.write(str(collectiveness.tolist())+"\n")
    f.write(str(conflict.tolist())+"\n")
    f.write(str(density.tolist())+"\n")
    f.write(str(uniformity.tolist())+"\n")
    
    f.close()
    # We get the name of all the files in the directory
    training_files = glob(path+"*.avi")
    for file_path in training_files:
        extract_descriptors(file_path,persp_map, out_file)
        print(file_path,": DONE",sep = "")

# Function to get the descriptors of a set of training videos and store them on a single file
def get_Test_Descriptors(path,persp_map, out_path = "./"):
    # We get the name of all the files in the directory
    test_files = glob(path+"*.avi")
    i = 1
    for file_path in test_files:
        name = file_path.split("/")[-1][:-4]
        # We reset the file if it exists
        f = open(out_path+name,"w")
        f.close()
        extract_descriptors(file_path,persp_map, out_path+name)
        i += 1
        print(file_path,": DONE",sep = "")

# Function to search for the maximun values of each descriptor on training
def get_Max_Descriptors(des_file, n_descriptors = 7):
    # Max value of every descriptor on training
    maximos = [0 for i in np.arange(n_descriptors)]
    i = 0
    f = open(des_file)
    for line in f:
        lista =  lista = np.fromstring(line[1:-2], dtype=float, sep=', ' )
        aux = max(lista)
        if aux > maximos[i]:
            maximos[i] = aux
        # Descriptors are stored in sequential order so we have to rotate in each iteration
        i = (i+1) % n_descriptors
        
    f.close()

    return maximos

################## ONE CLASS SVM #####################

# Function to get the normalized histograms of a set of descriptors
def get_Histograms(des_file, range_max, n_descriptors = 7):
    f = open(des_file)
    histograms = [[] for i in np.arange(n_descriptors)]
    i = 0
    for line in f:
        lista = np.fromstring(line[1:-2], dtype=float, sep=', ' )
        h = np.histogram(lista, bins = 16, range = (0,range_max[i]))[0]
        norm = np.linalg.norm(h)
        if norm != 0:
            histograms[i].append(h / np.linalg.norm(h))
        else:
            histograms[i].append(h)
        
        i = (i+1) % n_descriptors

    histograms = [np.concatenate(x) for x in zip(*histograms)]
    f.close()
        
    return histograms
    
def train_OC_SVM(samples, out_file = "svm.plk"):
    svm = OneClassSVM(nu =0.01, verbose = False, kernel = "sigmoid").fit(samples)
    joblib.dump(svm, out_file)

def test_OC_SVM(samples,in_file = "svm.plk"):
    svm = joblib.load(in_file)

    return svm.predict(samples)

################################################################

L = 20
t1 = -5
t2 = 5
min_motion = 0.01
fast_threshold = 20

escena = 1

path_p_map = "Datasets/UMN/Training/Escena "+str(escena)+"/persp_map.npy"
# if(os.path.isfile(path_p_map)):
#     p_map = np.load(path_p_map)
# else:
p_map = np.identity(3)

start = time.time()
get_Training_Descriptors("Datasets/UMN/Training/Escena "+str(escena)+"/",p_map, "Training Descriptors/escena"+str(escena)+"_tra_descriptors")
print("Tiempo extracción en training: {:4.5}".format(time.time()-start))

get_Test_Descriptors("Datasets/UMN/Test/Escena "+str(escena)+"/", p_map, "Full Video Descriptors/Escena "+str(escena)+"/")
print("Tiempo extracción en test: {:4.5}".format(time.time()-start))

gt = get_Ground_Truth("Datasets/UMN/ground_truth.txt")

range_max = get_Max_Descriptors("Training Descriptors/escena"+str(escena)+"_tra_descriptors")
hist = get_Histograms("Training Descriptors/escena"+str(escena)+"_tra_descriptors", range_max)

train_OC_SVM(hist)

total_labels = []
total_predicted = np.array([])
for d in glob("Full Video Descriptors/Escena "+str(escena)+"/*"):
    hist = get_Histograms(d, range_max)
    n_fr_train = gt[d.split("/")[-1]][0]//2
    begin_anom = n_fr_train
    end_anom = gt[d.split("/")[-1]][1] - n_fr_train
    labels = [1 if (i < (begin_anom-L) or i > (end_anom-L))  else -1 for i in np.arange(len(hist))]
    predicted = test_OC_SVM(hist)
    #print(predicted)
    
    score = accuracy_score(labels,predicted,normalize = False)
    print(d.split("/")[-1],score," - ",len(hist),"\t Prec: {:1.2f}".format(score/len(hist)))
    
    total_predicted = np.concatenate((total_predicted,predicted))
    total_labels += labels

print("Accuracy: {:1.3f}".format(accuracy_score(total_labels,total_predicted)))
print("AUC: {:1.3f}".format(roc_auc_score(total_labels,total_predicted)))

print("Tiempo total: {:4.5}".format(time.time()-start))

#L = 5
#mm = 0.05
# E1
#Accuracy: 0.948
#AUC: 0.953
# E2
#Accuracy: 0.796
#AUC: 0.814
# E3
#Accuracy: 
#AUC: 
