import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import product
from sklearn.decomposition import PCA

from data import get_Range_Descriptors
from data import prepare_Hist_and_Labels

from autoencoders import Autoencoder

def train_and_Test(training, test, video_classification, params_training, bins_vals, encoder_vals, verbose = 0, eliminar_descriptores = []):    
    acc_list = []
    auc_list = []
    params_list = []

    range_max, range_min = get_Range_Descriptors(training, video_classification)

    for n_bins in bins_vals:
        hist_original, labels = prepare_Hist_and_Labels(training, range_max,range_min, video_classification, n_bins, eliminar_descriptores, eliminar_vacios = True)
        
        hist_test_original, labels_test = prepare_Hist_and_Labels(test, range_max,range_min,video_classification, n_bins, eliminar_descriptores)

        for code_size in encoder_vals:
            if code_size is None:
                hist = hist_original.copy()
                hist_test = hist_test_original.copy()
                
            elif code_size < 1.0:
                pca = PCA(n_components=code_size)
                pca.fit(hist_original)
                hist = pca.transform(hist_original)
                hist_test = pca.transform(hist_test_original)
                
            else:
                autoencoder = Autoencoder(len(hist_original[0]), code_size)
                autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
                autoencoder.fit(hist_original,hist_original,verbose = verbose-1, epochs = 50)
                hist = autoencoder.encoder.predict(hist_original)
                hist_test = autoencoder.encoder.predict(hist_test_original)

            keys, values = zip(*params_training.items())
            permutations = [dict(zip(keys, v)) for v in product(*values)]

            for params in permutations:
                if "C" in params_training:
                    model = train_SVC(hist,labels, params)
                elif "OC" in params_training:
                    model = train_OC_SVM(hist,params)
                elif "hidden_layer_sizes" in params_training:
                    model = train_Network(hist, labels, params)
            

                prediction = test_model(hist_test, model, video_classification)

                acc = accuracy_score(labels_test,prediction)
                
                params_list.append(dict({"n_bins":n_bins,"code_size":code_size},**params))
                try:
                    auc = roc_auc_score(labels_test,prediction)
                except:
                    auc = 0

                acc_list.append(acc)
                auc_list.append(auc)
                    
                if verbose > 0:
                    print("ACC: {:1.2f} - AUC: {:1.2f} - {}".format(acc, auc, params_list[-1]))

    return acc_list, auc_list, params_list

#########################################################################################

def train_OC_SVM(samples, params, out_file = None):
    p = params.copy()
    p.pop("OC")
    svm = OneClassSVM(verbose = False, kernel = "sigmoid", **p).fit(samples)
    if out_file is not None:
        joblib.dump(svm, out_file)

    return svm

def train_SVC(samples, labels, params, out_file = None):
    svm = SVC(kernel = "rbf", decision_function_shape = 'ovo', class_weight = 'balanced', **params)
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
