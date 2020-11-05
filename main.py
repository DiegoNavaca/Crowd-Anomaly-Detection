import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import time
from collections import namedtuple
from glob import glob
import joblib
import numba
from numba import jit
from itertools import combinations

# Transforms a string into a list if it has the correct representation (faster than eval())
def string_To_List(s):
    lista = s[1:-2].split(", ")
    return [float(i) for i in lista]

# Reads the ground truth from a file
def get_Ground_Truth(in_file):
    f = open(in_file)
    gt = {}
    for line in f:
        lista = line.split(",")
        # name: beginning anomaly
        gt[lista[0]] = int(lista[1])

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

    return np.array(cliques)

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
def calculateMovement(features, trayectories, min_motion = 1.0):
    static_features = []
    velocity = np.array([np.linalg.norm(tracklet[0]-tracklet[-1]) for tracklet in trayectories])
    static_features = np.where(velocity < min_motion)[0]
    # # We calculate the total length of every trayectory
    # for i, (tracklet) in enumerate(trayectories):
    #     # #Motion of the entire trayectory
    #     # motion = 0
    #     # prev = tracklet[0]
    #     # for j in range(1,len(tracklet)):
    #     #     nex = tracklet[j]
    #     #     motion += euDistance(prev,nex)
    #     #     prev = nex

    #     # Distance between initial and final state
    #     motion = np.linalg.norm(tracklet[0]-tracklet[-1])

    #     # If the length is < beta we discard it
    #     if motion < min_motion:
    #         static_features.append(i)
    #     # else we save its velocity
    #     else:
    #         velocity.append(motion / len(tracklet))

    #We remove the static features
    features = np.delete(features,static_features,0)
    velocity = np.delete(velocity,static_features,0)
    trayectories = np.delete(trayectories,static_features,0)

    velocity = velocity / [len(tracklet) for tracklet in trayectories]

    return features, velocity, trayectories

def calculateDirectionVar(trayectories):
    direction_variation = np.array([np.sum([np.abs( direction(tracklet[i],tracklet[i+1]) - direction(tracklet[i+1],tracklet[i+2]) )
                                               for i in np.arange(len(tracklet)-2)]) / len(tracklet) for tracklet in trayectories])

    return direction_variation

def calculateStability(cliques, trayectories, t2 = -2):
    stability = np.zeros(len(trayectories))
    # For every tracklet
    for i in np.arange(len(trayectories)):
        # We calculate  the change of size and shape of all the posible triangles in the clique
        for pair in combinations(cliques[i],2):
            old_triangle = (trayectories[i][t2],trayectories[pair[0]][t2], trayectories[pair[1]][t2])
            new_triangle = (trayectories[i][-1],trayectories[pair[0]][-1], trayectories[pair[1]][-1])
            stability[i] += distanceTriangles(old_triangle, new_triangle)
        stability[i] = stability[i] / (len(cliques[i])+1)

    return stability

def calculateCollectiveness(cliques, trayectories):
    # Initialization
    collectiveness = [0 for vector in trayectories]
    # For every feature point
    for k in np.arange(len(cliques)):
        # We average the angular diference between the motion vector of every point with its neighbours
        for elem in cliques[k]:
            collectiveness[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[elem][0],trayectories[elem][-1]))
        collectiveness[k] /= (len(cliques[k])+1)

    return collectiveness

def calculateConflict(cliques, trayectories):
    # Initialization
    conflict = [0 for vector in trayectories]
    # For every feature point
    for k in np.arange(len(cliques)):
        # We average quotient of the angular diference between the motion vector of every point with its neighbours and their distances
        for elem in cliques[k]:
            conflict[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[elem][0],trayectories[elem][-1])) / np.linalg.norm(trayectories[k][-1] - trayectories[elem][-1])
        conflict[k] /= (len(cliques[k])+1)

    return conflict


def calculateCollectivenessAndConflict(cliques, trayectories):
    # Initialization
    collectiveness = np.zeros(len(trayectories))
    conflict = collectiveness.copy()
    # For every feature point
    for k in np.arange(len(cliques)):
        # We measure collectiveness and conflict
        for elem in cliques[k]:
            dif = difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[elem][0],trayectories[elem][-1]))
            collectiveness[k] += dif
            conflict[k] += dif / np.linalg.norm(trayectories[k][-1] - trayectories[elem][-1])
        collectiveness[k] /= (len(cliques[k])+1)
        conflict[k] /= (len(cliques[k])+1)

    return collectiveness, conflict

@jit(nopython = True)
def auxDensity(f1, f2, bandwidth):
    v =  np.exp(-1 * ( np.linalg.norm(f1 - f2) ) / 2*bandwidth**2)
    return v

def calculateDensity(cliques,features, bandwidth = 0.5):
    # Bandwidth = Bandwidth of the 2D Gaussian Kernel

    density = np.array([np.sum([auxDensity(features[i][0], features[elem][0], bandwidth) for elem in cliques[i]])
               / np.sqrt(2*np.pi)*bandwidth
                           for i in np.arange(len(cliques))])

    return density

def calculateUniformity(cliques, clusters, features):
    #Initialization
    #uniformity = [0 for i in np.arange(max(max(clusters)+1,1))]
    uniformity = np.zeros(max(max(clusters)+1,1))
    #inter_cluster = [0 for i in uniformity]
    inter_cluster = np.zeros(len(uniformity))
    intra_cluster = inter_cluster.copy()
    total_sum = 0
    #dist_matrix = [[-1 for f2 in features] for f1 in features]
    dist_matrix = np.full((len(features),len(features)),-1.0,dtype = float)

    # For every pair of point in each clique
    for f in np.arange(len(features)):
        for elem in cliques[f]:
            # We measure the distance if we have to
            if dist_matrix[f][elem] == -1:
                    dist_matrix[f][elem] = np.linalg.norm(features[f][0] - features[elem][0])
                    dist_matrix[elem][f] = dist_matrix[f][elem]
            # We follow the formula for the uniformity of each cluster
            if(f != elem):
                if(clusters[f] == clusters[elem]):
                    intra_cluster[clusters[f]] += 1 / dist_matrix[f][elem]
                else:
                    inter_cluster[clusters[f]] += 1 / dist_matrix[f][elem]
                total_sum += 1 / dist_matrix[f][elem]

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
    for i, (tracklet) in enumerate(trayectories):
        prev = tracklet[0]
        prev = (int(prev[0]),int(prev[1]))
        for i in np.anp.arange(1,len(tracklet)):
            nex = tracklet[i]
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
def extract_descriptors(video_file, out_file = "descriptors", L = 5, min_motion = 0.025, fast_threshold = 20):
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
        trayectories_aux.append([np.array((a,b))])

    while(video_open):
        it += 1

        if(it % L == 0):
            # Feature detection
            prev = prev_aux.copy()
            prev_key = fast.detect(prev_frame,None)
            prev_aux = cv.KeyPoint_convert(prev_key)
            prev_aux = prev_aux.reshape(-1, 1, 2)

            trayectories = trayectories_aux.copy()
            trayectories_aux = []
            for p in prev_aux:
                a, b = p.ravel()
                trayectories_aux.append([np.array((a,b))])

        # We calculate the metrics and begin a new set of trayectories every L frames
        if(it >= L):
            #Metrics analysis
            if len(trayectories) > 0:
                arr_trayectories = np.array(trayectories)
                prev, velocity, arr_trayectories = calculateMovement(prev,arr_trayectories, min_motion*L)
                trayectories = arr_trayectories.tolist()
            #vel_hist = np.histogram(velocity, bins = 16, range = (0,prev.shape[0]//L))[0] #

            if len(prev) > 2:

                dir_var = calculateDirectionVar(arr_trayectories)
                #dir_hist = np.histogram(dir_var, bins = 16, range = (0,3))[0]  #

                # Delaunay representation
                rect = (0, 0, prev_frame.shape[1], prev_frame.shape[0])
                delaunay.initDelaunay(rect)

                for point in prev:
                    a, b = point.ravel()
                    if(imgContains(frame,(a,b))):
                        delaunay.insert((a,b))

                cliques = getCliques(delaunay, prev)

                # Interactive Behaviours
                stability = calculateStability(cliques,arr_trayectories)

                #collectiveness = calculateCollectiveness(cliques,trayectories)

                #conflict = calculateConflict(cliques,trayectories)

                collectiveness, conflict = calculateCollectivenessAndConflict(cliques,arr_trayectories)

                density = calculateDensity(cliques,prev)

                clusters = getClusters(prev)
                uniformity = calculateUniformity(cliques, clusters, prev)
                #uniformity = [0,0]

                # Buscar una forma mejor
                file.write(str(velocity.tolist())+"\n")
                file.write(str(dir_var.tolist())+"\n")
                file.write(str(stability.tolist())+"\n")
                file.write(str(collectiveness.tolist())+"\n")
                file.write(str(conflict.tolist())+"\n")
                file.write(str(density.tolist())+"\n")
                file.write(str(uniformity.tolist())+"\n")


                # Image representation for checking results
                #addTrayectoriesToImage(trayectories,frame)
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
                    nex, status, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev, None)
                    aux, status, error = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex, prev)

                    # Selects good feature points for previous position
                    for i in np.arange(len(status)-1, -1, -1):
                        if status[i] == 0:
                            del trayectories[i]

                    # Selects good feature points for nex position
                    good_new = nex[status == 1]

                    for i, (new) in enumerate(good_new):
                        # Adds the new coordinates to the graph and the trayectories
                        a, b = new.ravel()
                        if(imgContains(frame,(a,b))):
                            trayectories[i].append(np.array((a,b)))
                            del trayectories[i][0]

                    # Updates previous good feature points
                    prev = good_new.reshape(-1, 1, 2)

            nex_aux, status_aux, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev_aux, None)
            aux, status_aux, error = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex_aux, prev_aux)

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
                    if(imgContains(frame,(a,b))):
                        trayectories_aux[i].append(np.array((a,b)))

                # Updates previous good feature points
                prev_aux = good_new.reshape(-1, 1, 2)

                # Updates previous frame
                prev_frame = frame.copy()

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
    file.close()

# Function to get the descriptors of a set of training videos and store them on a single file
def get_Training_Descriptors(path,out_file = "training_descriptors"):
    # We reset the file if it exists
    f = open(out_file,"w")
    f.close()
    # We get the name of all the files in the directory
    training_files = glob(path+"*")
    for file in training_files:
        extract_descriptors(file,out_file)
        print(file,": DONE",sep = "")

# Function to get the descriptors of a set of training videos and store them on a single file
def get_Test_Descriptors(path,out_path = "./"):
    # We get the name of all the files in the directory
    test_files = glob(path+"*")
    i = 1
    for file in test_files:
        name = file.split("/")[-1][:-4]
        # We reset the file if it exists
        f = open(out_path+name,"w")
        f.close()
        extract_descriptors(file,out_path+name)
        i += 1
        print(file,": DONE",sep = "")

# Function to search for the maximun values of each descriptor on training
def get_Max_Descriptors(des_file, n_descriptors = 7):
    # Max value of every descriptor on training
    maximos = [0 for i in np.arange(n_descriptors)]
    i = 0
    f = open(des_file)
    for line in f:
        lista = string_To_List(line)
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
        lista = string_To_List(line)
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
    svm = OneClassSVM(nu =0.01, kernel = "sigmoid", coef0 = 0.15).fit(samples)
    joblib.dump(svm, out_file)

def test_OC_SVM(samples,in_file = "svm.plk"):
    svm = joblib.load(in_file)

    return svm.predict(samples)

################################################################

escena = 1

start = time.time()
get_Training_Descriptors("Datasets/UMN/Training/Escena 1/","Training Descriptors/escena1_tra_descriptors")
print(time.time()-start)
#get_Training_Descriptors("Datasets/UMN/Training/Escena 2/","Training Descriptors/escena2_tra_descriptors")
#get_Training_Descriptors("Datasets/UMN/Training/Escena 3/","Training Descriptors/escena3_tra_descriptors")
get_Test_Descriptors("Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/","Full Video Descriptors/Escena "+str(escena)+"/")

gt = get_Ground_Truth("Datasets/UMN/ground_truth.txt")

range_max = get_Max_Descriptors("Training Descriptors/escena"+str(escena)+"_tra_descriptors")
hist = get_Histograms("Training Descriptors/escena"+str(escena)+"_tra_descriptors", range_max)
# range_max = get_Max_Descriptors("Training Descriptors/escena0_tra_descriptors")
# hist = get_Histograms("Training Descriptors/escena0_tra_descriptors", range_max)
print(time.time()-start)
train_OC_SVM(hist)

total_labels = []
total_predicted = np.array([])
for d in glob("Full Video Descriptors/Escena "+str(escena)+"/*"):
    hist = get_Histograms(d, range_max)
    limit = gt[d.split("/")[-1]]
    labels = [1 if i < limit else -1 for i in np.arange(len(hist))]
    predicted = test_OC_SVM(hist)

    score = accuracy_score(labels,predicted,normalize = False)
    print(d.split("/")[-1],score," - ",len(hist))

    total_predicted = np.concatenate((total_predicted,predicted))
    total_labels += labels

print("Accuracy:",accuracy_score(total_labels,total_predicted))
print("AUC:",roc_auc_score(total_labels,total_predicted))

print(time.time()-start)

#L = 5
#mm = 0.025
#772 - 805
#597 - 609
# E1
#Accuracy: 0.9681753889674681
#AUC: 0.978975625823452
# E3
#Accuracy: 0.951818611242324
#AUC: 0.9722070844686648

# No uniformity
# E1
# Accuracy: 0.9809052333804809
# AUC: 0.9882608695652174
# E2
# Accuracy: 0.8728564429201372
# AUC: 0.852110923910289
# E3
# Accuracy: 0.9754369390647142
# AUC: 0.9858310626702997
