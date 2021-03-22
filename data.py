import numpy as np
import pickle

from files import read_Labels

def get_Range_Descriptors(files, is_video_classification, n_descriptors = 8):
    # Max value of every descriptor on training
    maximos = np.zeros(n_descriptors)
    for des_file in files:
        labels = read_Labels(des_file+".labels")

        f = open(des_file+".data","rb")
        k = 0
        while True:
            try:
                descriptores = pickle.load(f)

                try:
                    if is_video_classification:
                        label = labels[0]
                    else:
                        label = labels[k]
                except IndexError:
                    label = 0

                # Anomaly values aren't taken into account
                if label != -1:
                    for i, d in enumerate(descriptores):
                        # We use percentile to remove the outliers
                        aux = np.percentile(d,95)
                    
                        if aux > maximos[i]:
                            maximos[i] = aux

                k += 1

            except EOFError:
                break
    
        f.close()

    return maximos, np.zeros(n_descriptors)

# Function to get the normalized histograms of a set of descriptors
def get_Histograms(des_file, range_max, range_min, n_bins, eliminar_descriptores):
    
    f = open(des_file+".data", "rb")
    histograms = [[] for i in np.arange(len(range_max))]
    vacios = []
    k = 0
    while True:
        try:
            descriptores = pickle.load(f)
            if len(descriptores[0]) > 1:
                for i, d in enumerate(descriptores):
                    
                    if i in eliminar_descriptores:
                        h = np.zeros(n_bins)
                    else:
                        h = np.histogram(d, bins = n_bins, range = (range_min[i],range_max[i]))[0]
                        
                    norm = np.linalg.norm(h)
                    
                    if norm != 0:
                        histograms[i].append(h/norm)
                    else:
                        histograms[i].append(h)
                        
            else:
                for i in range(len(descriptores)):
                    histograms[i].append(np.zeros(n_bins))
                vacios.append(k)

            k += 1
                
        except EOFError:
            break

    f.close()

    histograms = [np.concatenate(x) for x in zip(*histograms)]
        
    return histograms, vacios

def prepare_Hist_and_Labels(files, range_max,range_min, is_video_classification, n_bins, eliminar_descriptores, eliminar_vacios =False):
    histograms = []
    labels = []
    
    for f in files:
        if is_video_classification:
            h, vacios = get_Histograms(f, range_max, range_min, n_bins, eliminar_descriptores)

            # We remove frames without information to get a more reliable average
            for i in range(len(vacios)-1,0,-1):
                del h[vacios[i]] 

            # The values of the video are the average of the values in all frames
            h = [sum(x)/len(x) for x in zip(*h)]
            
            if len(h) != 0:
                histograms.append(h)
            
                labels += read_Labels(f+".labels")
            
        else:
            h, vacios = get_Histograms(f, range_max, range_min, n_bins, eliminar_descriptores)
            lab = read_Labels(f+".labels")

            if eliminar_vacios:
                for it in range(len(vacios)-1,0,-1):
                    del h[vacios[it]]
                    del lab[vacios[it]]
            
            histograms += h
            labels += lab
            
            

    return histograms, labels
 
