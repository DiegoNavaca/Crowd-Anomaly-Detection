import glob
import time
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from math import ceil

from files import extract_Descriptors_Dir
from files import get_Ground_Truth
from files import get_Classes
from SVM import train_and_Test_SVC

def try_Dataset(dataset, descriptors_dir, video_dir, params, tam_test, verbose = True, video_classification = False, C_vals = (5,10,25), bin_vals = (16,32)):
    
    gt = get_Ground_Truth("Datasets/"+dataset+"/ground_truth.txt")

    start = time.time()
    extract_Descriptors_Dir(params, video_dir, descriptors_dir,gt, verbose, video_classification, skip_extraction)
    if verbose:
        print(time.time()-start)

    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]

    best_acc = 0
    best_auc = 0
    best_C = 0.1
    n_folds = ceil(len(names)/tam_test)
    for n_bins in bin_vals:
        for C in C_vals:
            average_acc = 0
            average_auc = 0
            for i in range(n_folds):
                start = time.time()
                test = names[i*tam_test:i*tam_test+tam_test]
                training = names[:i*tam_test]+names[i*tam_test+tam_test:]

                prediction, labels = train_and_Test_SVC(training, test, C, video_classification, n_bins)
    
                acc = accuracy_score(labels,prediction)
                try:
                    auc = roc_auc_score(labels,prediction)
                except:
                    auc = 0
                    print(confusion_matrix(labels,prediction))

                if verbose:
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
                print("C: {}\tNº bins: {}".format(C,n_bins))
                print("Accuracy: {:1.3f}".format(average_acc))
                print("AUC: {:1.3f}\n".format(average_auc))

            if average_auc > best_auc:
                best_acc = average_acc
                best_auc = average_auc
                best_C = C
                best_n_bins = n_bins

    return best_acc, best_auc, best_C, best_n_bins

def try_Multiclass(dataset, descriptors_dir, video_dir, params, number_of_tests, verbose = True, C_vals = (5,10,25), bins_vals = (16,32), remove_same_scenes = True):
    
    classes = get_Classes("Datasets/"+dataset+"/ground_truth.txt")
    gt = get_Ground_Truth("Datasets/"+dataset+"/ground_truth.txt")
    
    if remove_same_scenes:
        scenes = get_Ground_Truth("Datasets/"+dataset+"/scenes.txt")

    extract_Descriptors_Dir(params, video_dir, descriptors_dir,gt, verbose, True, skip_extraction)

    best_acc = 0
    best_mat = None

    for n_bins in bins_vals:
        for C in C_vals:
            start = time.time()
            acc = 0
            conf_mat = np.zeros((len(classes),len(classes)))
            for n in range(number_of_tests):
            
                test = np.empty(len(classes),dtype = object)
                training = []
            
                for k, i in enumerate(classes):
                
                    indice_test = np.random.randint(len(classes[i]))
                
                    test[k] = classes[i][indice_test]
                    tr = classes[i][:indice_test]+classes[i][indice_test+1:]
                
                    if remove_same_scenes:
                        for j in range(len(tr)-1,0,-1):
                            if scenes[tr[j]] == scenes[test[k]]:
                                del tr[j]

                    training += tr

                test = [descriptors_dir+t for t in test]
                training = [descriptors_dir+t for t in training]

                prediction, labels = train_and_Test_SVC(training, test, C, True, n_bins)

                acc += accuracy_score(labels,prediction)
                try:
                    conf_mat += confusion_matrix(labels,prediction)
                except:
                    pass

                if verbose:
                    print("{}. Accuracy: {:1.3f}".format(n,acc/(n+1)))
                    print("Average Time: {:1.3f}".format((time.time()-start)/(n+1)))

            acc /= number_of_tests
            conf_mat /= number_of_tests

            if verbose:
                print("Total Validation Time: {:1.3f}".format(time.time()-start))
                print("C: {}\tNº bins: {}".format(C,n_bins))
                print("Accuracy: {:1.3f}".format(acc))
                print(conf_mat)

            if acc > best_acc:
                best_acc = acc
                best_mat = conf_mat
                best_C = C
                best_bins = n_bins

    return best_acc,best_mat,best_C, best_bins

def try_UMN(escena, params, verbose = True):
    descriptors_dir = "Descriptors/UMN/Escena "+str(escena)+"/"
    video_dir = "Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/"

    params["num_frames"] = -1

    acc, auc, C, n_bins = try_Dataset("UMN", descriptors_dir, video_dir, params, 1, verbose)

    print("RESULTADOS:")
    print("C: {}\tNº bins: {}".format(C,n_bins))
    print("Accuracy: {:1.3f}".format(acc))
    print("AUC: {:1.3f}".format(auc))

    return acc, auc, C, n_bins

def try_CVD(params, verbose = True):
    descriptors_dir = "Descriptors/CVD/"
    video_dir = "Datasets/Crowd Violence Detection/"

    #params["use_sift"] = 2000

    acc, auc, C, n_bins = try_Dataset("Crowd Violence Detection", descriptors_dir, video_dir, params, 50, verbose, video_classification = True, C_vals = (1,5,10,25))

    print("RESULTADOS:")
    print("C: {}\tNº bins: {}".format(C,n_bins))
    print("Accuracy: {:1.3f}".format(acc))
    print("AUC: {:1.3f}".format(auc))
    
    return acc, auc, C, n_bins

def try_CUHK(params, verbose = True):
    descriptors_dir = "Descriptors/CUHK/"
    video_dir = "Datasets/CUHK/Videos/"

    #params["others"]["skip_to_middle"] = True
    params["others"]["skip_frames"] = 1
    params["others"]["num_frames"] = 1
    params["others"]["change_resolution"] = 320

    acc, conf_mat, C, n_bins = try_Multiclass("CUHK", descriptors_dir, video_dir, params, 25, verbose, remove_same_scenes = False, C_vals = (8,16,32,64,128))

    print("RESULTADOS:")
    print("C: {}\tNº bins: {}".format(C,n_bins))
    print("Accuracy: {:1.3f}".format(acc))
    print("{}".format(conf_mat)) # row: class   column: prediction
    
    return acc, conf_mat, C, n_bins

#######################################################################

skip_extraction = False

params = {"L":30, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":20, "others":{}}
print(params)
#acc, auc, C, n_bins = try_UMN(2,params, verbose = True)
acc, auc, C, n_bins = try_CVD( params, verbose = True)
#acc, conf_mat, C, n_bins = try_CUHK( params, verbose = True)


################################ Resultados ################################

# Escena 1
# {'L': 15, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.992
# AUC: 0.990

# Escena 2
# {'L': 10, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# nu: 0.1
# Accuracy: 0.955
# AUC: 0.925

# Escena 3
# {'L': 20, 't1': -3, 't2': 2, 'min_motion': 0.05, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.983
# AUC: 0.973

# CVD
# {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":20, "others":{}} bins = 32
# C: 5
# AUC: 0.834
