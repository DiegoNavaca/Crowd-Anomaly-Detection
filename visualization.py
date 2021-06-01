import numpy as np
import cv2 as cv

########### IMAGE VISUALIZATION ###########

def imgContains(img,pt):
    return (pt[0] > 0 and pt[0] < img.shape[1] and pt[1] > 0 and pt[1] < img.shape[0])

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
    cv.circle(img,point,5,(0,0,155),4)
    for i in np.arange(len(cliques[index])):
        point = trayectories[cliques[index][i]][tr_index]
        point = (int(point[0]), int(point[1]))
        cv.circle(img,point,2,color,2)

def addClustersToImage(clusters, features, img):
    n_clusters = max(clusters)+1
    colores = [(np.random.randint(255),np.random.randint(255),np.random.randint(255))
                for i in range(n_clusters)]
    for i in np.arange(len(clusters)):
        point = features[i].ravel()
        point = (int(point[0]), int(point[1]))
        cv.circle(img,point,1,colores[clusters[i]],2)

def addPrediction(img,model, data, range_max, tam_border = 5):
    histograms = [np.histogram(data[i], bins = 16, range = (0,range_max[i]))[0] for i in range(len(range_max))]
    norms = [np.linalg.norm(h) for h in histograms]
    histograms = [histograms[i] / norms[i] if norms[i] != 0 else histograms[i] for i in range(len(histograms))]
    histograms = np.ravel(histograms).reshape(1,-1)
    
    prediction = model.predict(histograms)

    if prediction[0] == -1:
        img = cv.copyMakeBorder(img, tam_border, tam_border, tam_border, tam_border, cv.BORDER_CONSTANT, None, (0,0,255))

    return img
