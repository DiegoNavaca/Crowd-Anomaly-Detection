import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import product
from sklearn.decomposition import PCA

from keras.losses import MeanSquaredError
from keras.metrics import BinaryAccuracy
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

from autoencoders import Autoencoder
from data import get_Range_Descriptors
from data import prepare_Hist_and_Labels

def squeduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

def train_and_Test(training, test, video_classification, params, verbose = 0): 
    acc_list = []
    auc_list = []
    params_list = []

    range_max, range_min = get_Range_Descriptors(training, video_classification)

    for n_bins in params["bins"]:
        original_hist, original_labels = prepare_Hist_and_Labels(training, range_max,range_min, video_classification, n_bins, params.get("eliminar_descriptores",[]), eliminar_vacios = True, n_parts = params["n_parts"])
        
        original_test_hist, labels_test = prepare_Hist_and_Labels(test, range_max,range_min,video_classification, n_bins, params.get("eliminar_descriptores",[]), n_parts = params["n_parts"])

        for code_size in params["code_size"]:

            # Keep size of the features
            if code_size is None:
                hist = original_hist.copy()
                hist_test = original_test_hist.copy()

            # Use a PCA for dimensionality reduction
            elif code_size < 1.0:
                pca = PCA(n_components=code_size)
                #pca.fit([original_hist[i] for i in range(len(original_hist)) if original_labels[i] != -1])
                pca.fit(original_hist)
                hist = pca.transform(original_hist)
                hist_test = pca.transform(original_test_hist)

            # Use an autoencoder for dimensionality reduction
            else:
                labels = np.array([[1,0] if label == 1 else [0,1] for label in original_labels])
                autoencoder = Autoencoder(len(original_hist[0]), code_size, params["autoencoder"])
                class_loss = params["autoencoder"]["class_loss"]
                autoencoder.compile(optimizer = 'adam',
                                    loss = {"output_2":MeanSquaredError(),
                                            "output_1":class_loss},
                                    metrics = {"output_1":AUC(name='auc')})

                ES = EarlyStopping(monitor = 'val_output_1_loss',
                                            patience = 10, restore_best_weights = True)
                lrs = LearningRateScheduler(squeduler)
                
                history = autoencoder.fit(original_hist,{"output_2":original_hist,
                        "output_1":labels},verbose = 0, epochs = 100,
                        validation_split = 0.2, callbacks = [ES,lrs])
                if verbose > 0:
                    print("Autoencoder: AUCtrain: {:1.3f}\t AUCval {:1.3f}".format(
                        history.history['output_1_auc'][-1],history.history['val_output_1_auc'][-1]))
                hist = autoencoder.encoder.predict(original_hist)
                hist_test = autoencoder.encoder.predict(original_test_hist)

            keys, values = zip(*params["training"].items())
            param_permutations = [dict(zip(keys, v)) for v in product(*values)]

            for model_params in param_permutations:
                if "auto" in params["training"]:
                    model = autoencoder.classifier_model
                elif "C" in params["training"]:
                    model = train_SVC(hist,original_labels, model_params)
                elif "OC" in params["training"]:
                    model = train_OC_SVM(hist,model_params)
                elif "hidden_layer_sizes" in params["training"]:
                    model = train_Network(hist, original_labels, model_params)
                elif "n_estimators" in params["training"]:
                    model = train_RF(hist,original_labels, model_params)
            
                prediction = test_model(hist_test, model, video_classification)
                if "auto" in params["training"]:
                    prediction = [1 if p[0] > p[1] else -1 for p in prediction]

                acc = accuracy_score(labels_test,prediction)
                try:
                    auc = roc_auc_score(labels_test,prediction)
                except:
                    auc = acc
                
                params_list.append(dict({"n_bins":n_bins,"code_size":code_size},
                                        **model_params))
                
                acc_list.append(acc)
                auc_list.append(auc)
                    
                if verbose > 0:
                    print("ACC: {:1.2f} - AUC: {:1.2f} - {}".format(
                        acc, auc, params_list[-1]))

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

from sklearn.ensemble import RandomForestClassifier
def train_RF(samples, labels, params, out_file = None):
    model = RandomForestClassifier(**params)

    model.fit(samples,labels)

    if out_file is not None:
        joblib.dump(model, out_file)

    return model

def test_model(samples, model, video_classification):
    prediction = model.predict(samples)

    # If we don't have info about a frame (no person on scene) it's classified as normal 
    if not video_classification:    
        prediction = [1 if not samples[i].any() else prediction[i] for i in range(len(prediction))]

    return prediction
