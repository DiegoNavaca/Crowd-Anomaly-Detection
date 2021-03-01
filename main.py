import glob
import time
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from math import ceil

from files import extract_Descriptors_Dir
from files import get_Ground_Truth
from SVM import get_Max_Descriptors
from SVM import train_SVC
from SVM import test_SVM
from SVM import prepare_Hist_and_Labels

def try_Dataset(dataset, descriptors_dir, video_dir, params, tam_test, verbose = True, video_classification = False, nu_vals = (0.1,0.01,0.001)):
    
    gt = get_Ground_Truth("Datasets/"+dataset+"/ground_truth.txt")

    start = time.time()
    extract_Descriptors_Dir(params, video_dir, descriptors_dir,gt, verbose, video_classification)
    if verbose:
        print(time.time()-start)

    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]

    best_acc = 0
    best_auc = 0
    best_nu = 0.1
    n_folds = ceil(len(names)/tam_test)
    for nu in nu_vals:
        average_acc = 0
        average_auc = 0
        for i in range(n_folds):
            start = time.time()
            test = names[i*tam_test:i*tam_test+tam_test]
            training = names[:i*tam_test]+names[i*tam_test+tam_test:]
            
            range_max = get_Max_Descriptors(training, video_classification)
            print(range_max)

            hist, labels = prepare_Hist_and_Labels(training, range_max, video_classification, eliminar_vacios = True)
            print(len(hist), len(labels))

            model = train_SVC(hist, labels, nu = nu)

            hist, labels = prepare_Hist_and_Labels(test, range_max, video_classification)
            print(len(hist), len(labels))

            prediction = test_SVM(hist, range_max, model, video_classification)

            acc = accuracy_score(labels,prediction)
            try:
                auc = roc_auc_score(labels,prediction)
            except:
                auc = 0
                print(confusion_matrix(labels,prediction))

            if verbose:
                print("Tiempo: {}".format(time.time()-start))
                print("Nombre: {}".format(names[i]))
                print("Acertados: ",sum(1 for i in range(len(prediction)) if prediction[i] == labels[i]),"-",len(prediction))
                print("Positivos: {} - {}".format(
                    sum(1 for i in range(len(prediction)) if prediction[i] == labels[i] and prediction[i] == -1), labels.count(-1)))
                print("Accuracy: {:1.3f}".format(acc))
                print("AUC: {:1.3f}\n".format(auc))

            average_acc += acc
            average_auc += auc

        average_acc /= n_folds
        average_auc /= n_folds

        if verbose:
            print("nu: {}".format(nu))
            print("Accuracy: {:1.3f}".format(average_acc))
            print("AUC: {:1.3f}\n".format(average_auc))

        if average_auc > best_auc:
            best_acc = average_acc
            best_auc = average_auc
            best_nu = nu

    return best_acc, best_auc, best_nu

def try_UMN(escena, params, verbose = True):
    descriptors_dir = "Descriptors/UMN/Escena "+str(escena)+"/"
    video_dir = "Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/"

    acc, auc, nu = try_Dataset("UMN", descriptors_dir, video_dir, params, 1, verbose)

    return acc, auc, nu

def try_CVD(params, verbose = True):
    descriptors_dir = "Descriptors/CVD/"
    video_dir = "Datasets/Crowd Violence Detection/"

    acc, auc, nu = try_Dataset("Crowd Violence Detection", descriptors_dir, video_dir, params, 50, verbose, video_classification = True, nu_vals = (0.3,0.2,0.1))

    return acc, auc, nu

def try_CUHK(params, verbose = True):
    descriptors_dir = "Descriptors/CUHK/"
    video_dir = "Datasets/CUHK/Videos/"

    acc, auc, nu = try_Dataset("CUHK", descriptors_dir, video_dir, params, 95, verbose, video_classification = True, nu_vals = (0.3,0.2,0.1))
    return acc, auc, nu

# for L in (5,10,15,20):
# #L = 20
# #t1 = -5
#     for t1 in (-3,-5):
#         for t2 in (1,2):
#             for min_motion in (0.01,0.025, 0.05):
#                 params = {"L":L, "t1":t1, "t2":t2, "min_motion":min_motion, "fast_threshold":20}
#                 print(params)
#                 acc, auc, nu = try_UMN(2, params, verbose = False)
                
#                 print("nu: {}".format(nu))
#                 print("Accuracy: {:1.3f}".format(acc))
#                 print("AUC: {:1.3f}".format(auc))

params = {"L":5, "t1":-4, "t2":1, "min_motion":0.025 / (320+212), "fast_threshold":20, "max_num_features":2000}
print(params)
#acc, auc, nu = try_UMN(1,params, verbose = True)
acc, auc, nu = try_CVD( params, verbose = True)
#acc, auc, nu = try_CUHK( params, verbose = True)

print("nu: {}".format(nu))
print("Accuracy: {:1.3f}".format(acc))
print("AUC: {:1.3f}".format(auc))

# Escena 1
# {'L': 15, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.992
# AUC: 0.990

# Escena 2
# {'L': 10, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# nu: 0.1
# Accuracy: 0.955
# AUC: 0.923

# Escena 3
# {'L': 20, 't1': -3, 't2': 2, 'min_motion': 0.05, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.983
# AUC: 0.973

# CVD
# {'L': 5, 't1': -4, 't2': 1, 'min_motion': 4.699248120300752e-05, 'fast_threshold': 20, 'max_num_features': 1000}
# AUC: 0.8
