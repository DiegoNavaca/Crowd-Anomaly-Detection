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

# Transforms a string into a list if it has the correct representation (faster than eval())
def string_To_List(s):
    lista = s[1:-2].split(", ")
    return [float(i) for i in lista]

def imgContains(img,pt):
    return ( pt[0] >= 0 and pt[0] < img.shape[1] and pt[1] >= 0 and pt[1] < img.shape[0] )
    
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

# Calculates the area of a triangle (x2 for a better efficiency)
def areaTriangle(pt0,pt1,pt2):
    return ( pt0[0] * (pt1[1] - pt2[1]) + pt1[0] * (pt2[1] - pt0[1]) + pt2[0] * (pt0[1] - pt1[0]) )
    
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
    edges = grafo.getEdgeList()
    point = namedtuple("point", ["x", "y"])
    points = {point(features[i][0][0],features[i][0][1]):i for i in range(len(features))}        
    cliques = [[p] for p in range(len(features))]
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
    points = frozenset(point(grid[i][0][0],grid[i][0][1]) for i in range(len(grid)))
    add = [1 for f in new_features]
    for k in range(len(new_features)):
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

def calculateStability(cliques, trayectories, t2 = -2):
    stability = [0 for f in trayectories]
    # For every tracklet
    for i in range(len(trayectories)):
        # We calculate  the change of size and shape of all the posible triangles in the clique 
        for j in range(1,len(cliques[i])):
            for k in range(j+1,len(cliques[i])):
                old_triangle = (trayectories[i][t2],trayectories[cliques[i][j]][t2], trayectories[cliques[i][k]][t2])
                new_triangle = (trayectories[i][-1],trayectories[cliques[i][j]][-1], trayectories[cliques[i][k]][-1])
                stability[i] += distanceTriangles(old_triangle, new_triangle)
        stability[i] /= len(cliques[i])

    return stability
            

def calculateCollectiveness(cliques, trayectories):
    collectiveness = [sum(
        difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1])) for i in range(1,len(cliques[k]))
    ) / len(cliques[k]) for k in range(len(cliques))]
    
    # # Initialization
    # collectiveness = [0 for vector in trayectories]
    # # For every feature point
    # for k in range(len(cliques)):
    #     # We average the angular diference between the motion vector of every point with its neighbours
    #     for i in range(1,len(cliques[k])):
    #         collectiveness[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1]))
    #     collectiveness[k] /= len(cliques[k])

    return collectiveness

def calculateConflict(cliques, trayectories):
    conflict = [sum(
    difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1]))
        / euDistance(trayectories[k][-1], trayectories[cliques[k][i]][-1]) for i in range(1,len(cliques[k]))
    ) / len(cliques[k]) for k in range(len(cliques))]
   
    # # Initialization
    # conflict = [0 for vector in trayectories]
    # # For every feature point
    # for k in range(len(cliques)):
    #     # We average quotient of the angular diference between the motion vector of every point with its neighbours and their distances
    #     for i in range(1,len(cliques[k])):
    #         conflict[k] += difAng((trayectories[k][0],trayectories[k][-1]),(trayectories[cliques[k][i]][0],trayectories[cliques[k][i]][-1])) / euDistance(trayectories[k][-1], trayectories[cliques[k][i]][-1])
    #     conflict[k] /= len(cliques[k])

    return conflict

def calculateDensity(cliques,features, bandwidth = 0.5):
    # Bandwidth = Bandwidth of the 2D Gaussian Kernel
    n_features = len(cliques)

    density = [sum( np.exp(-1 * ( euDistance(features[i][0], features[cliques[i][j]][0]) ) / 2*bandwidth**2)
        for j in range(1,len(cliques[i]))
    ) / np.sqrt(2*np.pi)*bandwidth for i in range(n_features)]
    
    # density = [0 for i in range(n_features)]
    # for i in range(n_features):
    #     for j in range(1,len(cliques[i])):
    #         density[i] += np.exp(-1 * ( euDistance(features[i][0], features[cliques[i][j]][0]) ) / 2*bandwidth**2)
    #     density[i] /= np.sqrt(2*np.pi)*bandwidth

    return density

def calculateUniformity(cliques, clusters, features):
    #Initialization 
    uniformity = [0 for i in range(max(max(clusters)+1,1))]
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
            if(f != cliques[f][q]):
                if(clusters[f] == clusters[cliques[f][q]]):
                    intra_cluster[clusters[f]] += 1 / dist_matrix[f][cliques[f][q]]
                else:
                    inter_cluster[clusters[f]] += 1 / dist_matrix[f][cliques[f][q]]
                total_sum += 1 / dist_matrix[f][cliques[f][q]]

    uniformity = (intra_cluster / total_sum) - (inter_cluster / total_sum)**2

    # We return a list to keep consistency with the rest of the descriptors
    return list(uniformity)
    
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
        trayectories_aux.append([(a,b)])
    
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
                trayectories_aux.append([(a,b)])
        
        # We calculate the metrics and begin a new set of trayectories every L frames
        if(it >= L):
            #Metrics analysis
            # Individual Behaviours
            prev, velocity = calculateMovement(prev,trayectories, min_motion*L)
            #vel_hist = np.histogram(velocity, bins = 16, range = (0,prev.shape[0]//L))[0] #
        
            dir_var = calculateDirectionVar(trayectories)
            #dir_hist = np.histogram(dir_var, bins = 16, range = (0,3))[0]  #

            if len(prev) > 2:
                # Delaunay representation
                rect = (0, 0, prev_frame.shape[1], prev_frame.shape[0])
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
                
                
                file.write(str(velocity)+"\n")
                file.write(str(dir_var)+"\n")
                file.write(str(stability)+"\n")
                file.write(str(collectiveness)+"\n")
                file.write(str(conflict)+"\n")
                file.write(str(density)+"\n")
                file.write(str(uniformity)+"\n")
                

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
            # # Calculates sparse optical flow by Lucas-Kanade method
            # # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
            if (it > L) :
                if len(prev) > 0:
                    nex, status, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev, None)
                    aux, status, error = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex, prev)
        
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
                            del trayectories[i][0]                            
    
                    # Updates previous good feature points
                    prev = good_new.reshape(-1, 1, 2)

            nex_aux, status_aux, error = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev_aux, None)
            aux, status_aux, error = cv.calcOpticalFlowPyrLK(frame, prev_frame, nex_aux, prev_aux)

            if status_aux is not None:
        
                # Selects good feature points for previous position
                good_old = prev_aux[status_aux == 1]
                for i in range(len(status_aux)-1, -1, -1):
                    if status_aux[i] == 0:
                        del trayectories_aux[i]
                
                # Selects good feature points for nex position
                good_new = nex_aux[status_aux == 1]
        
                for i, (new) in enumerate(good_new):
                    # Adds the new coordinates to the graph and the trayectories
                    a, b = new.ravel()
                    if(imgContains(frame,(a,b))):
                        trayectories_aux[i].append((a,b))
        
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
        # We reset the file if it exists
        f = open(out_path+str(i),"w")
        f.close()
        extract_descriptors(file,out_path+str(i))
        i += 1
        print(file,": DONE",sep = "")

# Function to search for the maximun values of each descriptor on training 
def get_Max_Descriptors(des_file, n_descriptors = 7):
    # Max value of every descriptor on training
    maximos = [0 for i in range(n_descriptors)]
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
    histograms = [[] for i in range(n_descriptors)]
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
        
    return histograms
    
def train_OC_SVM(samples, out_file = "svm.plk"):
    svm = OneClassSVM(nu =0.01, kernel = "sigmoid", coef0 = 0.15).fit(samples)
    joblib.dump(svm, out_file)

def test_OC_SVM(samples,in_file = "svm.plk"):
    svm = joblib.load(in_file)

    return svm.predict(samples)

################################################################
    

np.random.seed(5)

start = time.time()
get_Training_Descriptors("Datasets/UMN/Training/Escena 1/","Training Descriptors/escena1_tra_descriptors")
#get_Training_Descriptors("Datasets/UMN/Training/Escena 2/","Training Descriptors/escena2_tra_descriptors")
#get_Training_Descriptors("Datasets/UMN/Training/Escena 3/","Training Descriptors/escena3_tra_descriptors")
get_Test_Descriptors("Datasets/UMN/Escenas Completas/Escena 1/","Full Video Descriptors/Escena 1/")

range_max = get_Max_Descriptors("Training Descriptors/escena1_tra_descriptors")
hist = get_Histograms("Training Descriptors/escena1_tra_descriptors", range_max)
print(time.time()-start)
train_OC_SVM(hist)

hist = get_Histograms("Full Video Descriptors/Escena 1/1", range_max)
labels = [1 if i < 666 else -1 for i in range(len(hist))]
predicted_1 = test_OC_SVM(hist)
score = accuracy_score(labels,predicted_1,normalize = False)
all_labels = labels.copy()

print(score,"-",len(hist))

hist = get_Histograms("Full Video Descriptors/Escena 1/2", range_max)
labels = [1 if i < 484 else -1 for i in range(len(hist))]
predicted_2 = test_OC_SVM(hist)
score = accuracy_score(labels,predicted_2,normalize = False)
all_labels += labels

print(score,"-",len(hist))

all_predictions = np.concatenate((predicted_1,predicted_2))

print("Accuracy:",accuracy_score(all_labels,all_predictions))
print("AUC:",roc_auc_score(all_labels,all_predictions))

print(time.time()-start)

#L = 5
#mm = 0.025
#772 - 805
#597 - 609
#Accuracy: 0.9681753889674681
#AUC: 0.9775164690382082

