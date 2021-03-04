import joblib
import pickle
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

        f = open(des_file+".data","rb")
        k = 0
        while True:
            try:
                descriptores = pickle.load(f)

                if video_classification:
                    label = labels[0]
                else:
                    label = labels[k]

                if label != -1:
                    for i, d in enumerate(descriptores):
                        aux = max(d)
                    
                        if aux > maximos[i]:
                            maximos[i] = aux

                k += 1

            except:
                break
    
        f.close()

    return maximos

# Function to get the normalized histograms of a set of descriptors
def get_Histograms(des_file, range_max):
    f = open(des_file+".data", "rb")
    histograms = [[] for i in np.arange(len(range_max))]
    vacios = []
    k = 0
    while True:
        try:
            descriptores = pickle.load(f)
            if len(descriptores[0]) > 1:
                for i, d in enumerate(descriptores):
                    h = np.histogram(d, bins = 16, range = (0,range_max[i]))[0]
                    norm = np.linalg.norm(h)
                    
                    if norm != 0:
                        histograms[i].append(h/norm)
                    else:
                        histograms[i].append(h)
                        
            else:
                for i in range(len(descriptores)):
                    histograms[i].append(np.zeros(16))
                vacios.append(k)

            k += 1
                
        except:
                break

    f.close()

    histograms = [np.concatenate(x) for x in zip(*histograms)]
        
    return histograms, vacios

def prepare_Hist_and_Labels(files, range_max, video_classification, eliminar_vacios =False):
    hist = []
    labels = []
    
    for f in files:
        if not video_classification:
            h, vacios = get_Histograms(f, range_max)
            
            hist += h
            labels += read_Labels(f+".labels")
            
            if eliminar_vacios:
                for i in range(len(vacios)-1,0,-1):
                    del hist[vacios[i]]
                    del labels[vacios[i]]
            
        else:
            h, vacios = get_Histograms(f, range_max)
            
            for i in range(len(vacios)-1,0,-1):
                del h[vacios[i]] 
                    
            #h = [sum(x)/len(x) for x in zip(*h)]
            h = [sum(x) for x in zip(*h)]
            hist.append(h)
            
            labels += read_Labels(f+".labels")

    return hist, labels

def train_and_Test_SVC(training, test, C, video_classification):
    range_max = get_Max_Descriptors(training, video_classification)

    hist, labels = prepare_Hist_and_Labels(training, range_max, video_classification, eliminar_vacios = True)
            
    model = train_SVC(hist, labels, C = C)
            
    hist, labels = prepare_Hist_and_Labels(test, range_max, video_classification)
        
    prediction = test_SVM(hist, range_max, model, video_classification)

    return prediction, labels

def train_OC_SVM(samples, out_file = None, nu = 0.1):
    svm = OneClassSVM(nu = nu, verbose = False, kernel = "sigmoid").fit(samples)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm

def train_SVC(samples, labels, out_file = None, C = 10):
    svm = SVC(C = C, kernel = "rbf", decision_function_shape = 'ovo', class_weight = 'balanced')
    svm.fit(samples, labels)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm 

def test_SVM(samples, range_max, model, video_classification):
    prediction = model.predict(samples)

    if not video_classification:    
        prediction = [1 if not samples[i].any() else prediction[i] for i in range(len(prediction))]

    return prediction
