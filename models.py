import joblib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import product
from sklearn.decomposition import PCA


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

                if is_video_classification:
                    label = labels[0]
                else:
                    label = labels[k]

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

def train_and_Test(training, test, video_classification, params_training, bins_vals, verbose = 0, eliminar_descriptores = []):    
    acc_list = []
    auc_list = []
    params_list = []

    range_max, range_min = get_Range_Descriptors(training, video_classification)

    for n_bins in bins_vals:
        hist, labels = prepare_Hist_and_Labels(training, range_max,range_min,
                                               video_classification, n_bins, eliminar_descriptores, eliminar_vacios = True)
        
        hist_test, labels_test = prepare_Hist_and_Labels(test, range_max,range_min,
                                                         video_classification, n_bins, eliminar_descriptores)

        #print(len([i for i in labels_test if i == 1]),"-", len(labels_test))
        
        pca = PCA(n_components=0.95)
        pca.fit(hist)
        hist = pca.transform(hist)
        hist_test = pca.transform(hist_test)

        if "hidden_layer_sizes" in params_training:
            keys, values = zip(*params_training.items())
            permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
            for params in permutations_dicts:
                model = train_Network(hist, labels, params)

                prediction = test_model(hist_test, model, video_classification)

                acc = accuracy_score(labels_test,prediction)
                
                params_list.append(dict({"n_bins":n_bins},**params))
                try:
                    auc = roc_auc_score(labels_test,prediction)
                except:
                    auc = 0

                acc_list.append(acc)
                auc_list.append(auc)
                    
                if verbose > 0:
                    print("ACC: {:1.2f} - AUC: {:1.2f} - {}".format(acc, auc, params))
            
        elif "C" in params_training:
            for C in params_training["C"]:
                model = train_SVC(hist,labels,C)

                prediction = test_model(hist_test, model, video_classification)

                acc = accuracy_score(labels_test,prediction)
                
                params_list.append({"n_bins":n_bins,"C":C})
                try:
                    auc = roc_auc_score(labels_test,prediction)
                except:
                    auc = 0

                acc_list.append(acc)
                auc_list.append(auc)
                    
                if verbose > 0:
                    print("ACC: {:1.2f} - AUC: {:1.2f} - {}".format(acc, auc, params_list[-1]))

    return acc_list, auc_list, params_list

def train_OC_SVM(samples, nu = 0.1, out_file = None):
    svm = OneClassSVM(nu = nu, verbose = False, kernel = "sigmoid").fit(samples)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm

def train_SVC(samples, labels, C = 10, out_file = None):
    svm = SVC(C = C, kernel = "rbf", decision_function_shape = 'ovo', class_weight = 'balanced')
    svm.fit(samples, labels)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm
    

def train_Network(samples, labels, params, out_file = None):
    model = MLPClassifier(max_iter = 2000, **params)

    model.fit(samples,labels)

    if out_file is not None:
        joblib.dump(model, out_file)

    return model

def test_model(samples, model, video_classification):
    prediction = model.predict(samples)

    if not video_classification:    
        prediction = [1 if not samples[i].any() else prediction[i] for i in range(len(prediction))]

    return prediction
