import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.svm import NuSVC
from sklearn.svm import SVC

from files import read_Labels

def get_Max_Descriptors(files, video_classification, n_descriptors = 8):
    # Max value of every descriptor on training
    maximos = np.zeros(n_descriptors)
    for des_file in files:
        labels = read_Labels(des_file+".labels")
    
        i = 0
        f = open(des_file+".data")
        for line in f:
            lista = np.fromstring(line[1:-1], dtype=float, sep=', ' )
            aux = max(lista)
            
            if video_classification:
                label = labels[0]
            else:
                label = labels[i//n_descriptors]
                
            if aux > maximos[i%n_descriptors] and label == 1:
                maximos[i%n_descriptors] = aux
            # Descriptors are stored in sequential order so we have to rotate in each iteration
            i += 1
        f.close()

    return maximos

# Function to get the normalized histograms of a set of descriptors
def get_Histograms(des_file, range_max):
    f = open(des_file+".data")
    histograms = [[] for i in np.arange(len(range_max))]
    i = 0
    for line in f:
        lista = np.fromstring(line[1:-1], dtype=float, sep=', ' )
        h = np.histogram(lista, bins = 16, range = (0,range_max[i]))[0]
        norm = np.linalg.norm(h)
        if norm != 0:
            histograms[i].append(h / norm)
        else:
            histograms[i].append(h)

        i = (i+1) % len(range_max)

    f.close()

    histograms = [np.concatenate(x) for x in zip(*histograms)]
        
    return histograms

def prepare_Hist_and_Labels(files, range_max, video_classification):
    hist = []
    labels = []
    
    for f in files:
        if not video_classification:
            hist += get_Histograms(f, range_max)
            
        else:
            h = get_Histograms(f, range_max)
            #h = [sum(x)/len(x) for x in zip(*h)]
            h = [sum(x) for x in zip(*h)]
            hist.append(h)
            
        labels += read_Labels(f+".labels")

    return hist, labels

def train_OC_SVM(samples, out_file = None, nu = 0.1):
    svm = OneClassSVM(nu = nu, verbose = False, kernel = "sigmoid").fit(samples)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm

def train_SVC(samples, labels, out_file = None, nu = 0.1):
    svm = NuSVC(nu = nu, kernel = 'rbf')
    try:
        svm.fit(samples, labels)
    except:
        print("Nu infeasible, using SVC instead")
        svm = SVC(C = 1/nu, kernel = "rbf")
        svm.fit(samples, labels)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm 

def test_SVM(samples, range_max, model):
    empty = np.concatenate([np.histogram(np.zeros(1),bins = 16, range = (0,i))[0] for i in range_max])

    prediction = model.predict(samples)
    
    prediction = [1 if (samples[i] == empty).all() else prediction[i] for i in range(len(prediction))]

    return prediction
