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
#from keras.metrics import BinaryAccuracy
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

from autoencoders import Autoencoder
from data import get_Range_Descriptors
from data import prepare_Hist_and_Labels

def squeduler(epoch, lr):
    if epoch < 10:
        return lr
    return lr * 0.9

def train_and_Test(training, test, is_video_classification, params, verbose = 0):
    acc_list = []
    auc_list = []
    params_list = []

    # We calculate the ranges of the histograms 
    range_max, range_min = get_Range_Descriptors(training, is_video_classification)

    for n_bins in params["bins"]:
        hist, labels = prepare_Hist_and_Labels(training, range_max,range_min, is_video_classification, n_bins, params.get("eliminar_descriptores",[]), eliminar_vacios = True, n_parts = params["n_parts"])
        
        test_hist, labels_test = prepare_Hist_and_Labels(test, range_max,range_min,is_video_classification, n_bins, params.get("eliminar_descriptores",[]), n_parts = params["n_parts"])

        for code_size in params["code_size"]:
            # Not use dimmensionality reduction
            if code_size == None:
                code = hist.copy()
                test_code = test_hist.copy()
            # Use a PCA for dimensionality reduction
            elif code_size < 1.0:
                pca = PCA(n_components=code_size)
                pca.fit(hist)
                code = pca.transform(hist)
                test_code = pca.transform(test_hist)
            # Use an autoencoder for dimensionality reduction
            else:
                coder_labels = np.array([[1,0] if l == 1 else [0,1] for l in labels])
                
                # Model definition and compilation
                autoencoder = Autoencoder(len(hist[0]), code_size, params["autoencoder"])
                class_loss = params["autoencoder"]["class_loss"]
                autoencoder.compile(optimizer = 'adam',
                                    loss = {"output_2":MeanSquaredError(),
                                            "output_1":class_loss},
                                    metrics = {"output_1":AUC(name='auc')})

                ES = EarlyStopping(monitor = 'val_output_2_loss', # CAMBIAR A 1
                                            patience = 10, restore_best_weights = True)
                lrs = LearningRateScheduler(squeduler)

                # We train the model
                history = autoencoder.fit(hist,{"output_2":hist,
                        "output_1":coder_labels},verbose = 0, epochs = 100,
                        validation_split = 0.2, callbacks = [ES,lrs])
                
                if verbose > 0:
                    print("Autoencoder: AUCtrain: {:1.3f}\t AUCval {:1.3f}".format(
                        history.history['output_1_auc'][-1],
                        history.history['val_output_1_auc'][-1]))

                # We apply the model to encode the inputs
                code = autoencoder.encoder.predict(hist)
                test_code = autoencoder.encoder.predict(test_hist)

            # We calculate all the permutations of parameters for the classifier
            keys, values = zip(*params["training"].items())
            param_permutations = [dict(zip(keys, v)) for v in product(*values)]

            # For each permutation we train and test the model
            for model_params in param_permutations:
                if "auto" in params["training"]:
                    model = autoencoder.classifier_model
                elif "C" in params["training"]:
                    model = train_SVC(code, labels, model_params)
                elif "OC" in params["training"]:
                    model = train_OC_SVM(code, model_params)
                elif "hidden_layer_sizes" in params["training"]:
                    model = train_Network(code, labels, model_params)

                prediction = test_model(test_code, model, is_video_classification)

                # If we use the autoencoder classifier we have to translate the labels
                if "auto" in params["training"]:
                    prediction = [1 if p[0] > p[1] else -1 for p in prediction]

                acc = accuracy_score(labels_test, prediction)
                auc = roc_auc_score(labels_test, prediction)

                # We save the result and the parameters used
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

    svm = OneClassSVM(verbose = False, **p).fit(samples)

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

def test_model(samples, model, is_video_classification):
    prediction = model.predict(samples)

    # If we don't have info about a frame (no person on scene) it's classified as normal
    if not is_video_classification:
        prediction = [1 if not samples[i].any() else prediction[i] for i in range(len(prediction))]

    return prediction
