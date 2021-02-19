import joblib
from sklearn.svm import OneClassSVM
from sklearn.svm import NuSVC

from utils import np
from files import read_Labels

def get_Max_Descriptors(files, n_descriptors = 7):
    # Max value of every descriptor on training
    maximos = [0 for i in np.arange(n_descriptors)]
    for des_file in files:
        labels = read_Labels(des_file+".labels")
    
        i = 0
        f = open(des_file+".data")
        for line in f:
            lista = np.fromstring(line[1:-2], dtype=float, sep=', ' )
            aux = max(lista)
            if aux > maximos[i%n_descriptors] and labels[i//n_descriptors] == 1:
                maximos[i%n_descriptors] = aux
            # Descriptors are stored in sequential order so we have to rotate in each iteration
            i = i+1
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
            histograms[i].append(h / np.linalg.norm(h))
        else:
            histograms[i].append(h)

        i = (i+1) % len(range_max)

    histograms = [np.concatenate(x) for x in zip(*histograms)]
    f.close()
        
    return histograms

def prepare_Hist_and_Labels(files, range_max):
    hist = []
    labels = []
    for f in files:
        h = get_Histograms(f, range_max)
        hist = hist+h
        
        lab = read_Labels(f+".labels")
        labels = labels+lab

    return hist, labels

def train_OC_SVM(samples, out_file = None):
    svm = OneClassSVM(nu =0.01, verbose = False, kernel = "sigmoid").fit(samples)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm

def train_SVC(samples, labels, out_file = None, nu = 0.1):
    svm = NuSVC(nu = nu, kernel = 'rbf')
    svm.fit(samples, labels)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm 

def test_SVM(samples, range_max, model):
    empty = np.concatenate([np.histogram(np.zeros(1),bins = 16, range = (0,i))[0] for i in range_max])

    prediction = model.predict(samples)
    
    prediction = [1 if (samples[i] == empty).all() else prediction[i] for i in range(len(prediction))]

    return prediction
