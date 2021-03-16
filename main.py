import glob
import time
import numpy as np

from sklearn.metrics import confusion_matrix
from math import ceil
from operator import mul
from functools import reduce

from files import extract_Descriptors_Dir
from files import get_Ground_Truth
from files import get_Classes
from models import train_and_Test

def try_Dataset(dataset, descriptors_dir, video_dir, params_extraction, params_training, tam_test, verbose = 2, is_video_classification = False):
    if verbose:
        verbose -= 1
    
    gt = get_Ground_Truth("Datasets/"+dataset+"/ground_truth.txt")

    start = time.time()
    extract_Descriptors_Dir(params_extraction, video_dir, descriptors_dir,gt, verbose, is_video_classification, skip_extraction)
    if verbose:
        print(time.time()-start)

    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]

    n_folds = ceil(len(names)/tam_test)
    n_pruebas = reduce(mul, [len(params_training[x]) for x in params_training], 1)
    
    bins_vals = params_training["bins"]
    params_training.pop("bins",None)
    
    average_acc = np.zeros(n_pruebas)
    average_auc = np.zeros(n_pruebas)
    for i in range(n_folds):
        start = time.time()
        test = names[i*tam_test:i*tam_test+tam_test]
        training = names[:i*tam_test]+names[i*tam_test+tam_test:]

        acc, auc, list_params = train_and_Test(training, test, is_video_classification, params_training, bins_vals)
        
        if verbose:
            #print("ACC:",tuple(zip(list_params,acc)))
            print("Time:",time.time()-start)
            if verbose > 1:
                for j in range(len(acc)):
                    print("ACC: {:1.2f} - AUC: {:1.2f} - {}".format(acc[j], auc[j], list_params[j]))
            best = np.argmax(auc)
            print("Best:\nParams: {}\n ACC: {:1.3f}\t AUC: {:1.3f}".format(list_params[best],acc[best], auc[best] ))
            print("Max Accuracy: {:1.3f}".format(max(acc)))
            print("Max AUC: {:1.3f}\n".format(max(auc)))
            
        average_acc = average_acc + np.array(acc)
        average_auc = average_auc + np.array(auc)

    average_acc /= n_folds
    average_auc /= n_folds

    best = np.argmax(average_auc)
    final_acc = average_acc[best]
    final_auc = average_auc[best]
    final_params = list_params[best]

    return final_acc, final_auc, final_params

def try_UMN(escena, params_extraction, params_training, verbose = 2):
    descriptors_dir = "Descriptors/UMN/Escena "+str(escena)+"/"
    video_dir = "Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/"

    acc, auc, best_params = try_Dataset("UMN", descriptors_dir, video_dir, params_extraction,params_training, 1, verbose)

    if verbose:
        print("RESULTADOS:")
        print(best_params)
        print("Accuracy: {:1.3f}".format(acc))
        print("AUC: {:1.3f}".format(auc))

    return acc, auc, best_params

def try_CVD(params_extraction,params_training, verbose = 2):
    descriptors_dir = "Descriptors/CVD/"
    video_dir = "Datasets/Crowd Violence Detection/"

    #params["use_sift"] = 2000

    acc, auc, best_params = try_Dataset("Crowd Violence Detection", descriptors_dir, video_dir, params_extraction, params_training, 50, verbose, is_video_classification = True)

    if verbose:
        print("RESULTADOS:")
        print(best_params)
        print("Accuracy: {:1.3f}".format(acc))
        print("AUC: {:1.3f}".format(auc))
    
    return acc, auc, best_params

# def try_CUHK(params, verbose = 2):
#     descriptors_dir = "Descriptors/CUHK/"
#     video_dir = "Datasets/CUHK/Videos/"

#     #params["others"]["skip_to_middle"] = True
#     params["others"]["skip_frames"] = 1
#     params["others"]["num_frames"] = 1
#     params["others"]["change_resolution"] = 320

#     acc, conf_mat, C, n_bins = try_Multiclass("CUHK", descriptors_dir, video_dir, params, 25, verbose, remove_same_scenes = False, C_vals = (8,16,32,64,128))

#     if verbose:
#         print("RESULTADOS:")
#         print("C: {}\tNº bins: {}".format(C,n_bins))
#         print("Accuracy: {:1.3f}".format(acc))
#         print("{}".format(conf_mat)) # row: class   column: prediction
    
#     return acc, conf_mat, C, n_bins

#######################################################################

skip_extraction = True

# results_file = open("results.txt","w")

# for L in [5,10]:
#     for t1 in [-3,-5]:
#         for t2 in [1,2]:
#             for min_motion in [0.025,0.05]:
#                 for fast_threshold in [10,20]:
#                     params = {"L":L, "t1":t1, "t2":t2, "min_motion":min_motion, "fast_threshold":fast_threshold, "others":{}}
#                     results_file.write(str(params)+"\n")
#                     results_file.flush()
#                     start = time.time()
#                     acc, auc, C, n_bins = try_CVD( params, verbose = 2)
#                     results_file.write("Acc: "+str(acc)+" AUC: "+str(auc)+" C: "+str(C)+" Nº bins: "+str(n_bins)+" Time: "+str(time.time()-start)+"\n")
#                     results_file.flush()

# results_file.close()

params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
#params_training = {"C":[8,16,32], "bins":[16,32]}
params_training = {"bins":[32], "hidden_layer_sizes":[(16,4),(64,8)], "solver":["adam", "lbfgs"],"alpha":[0.001,0.0001]}
#acc, auc, best_params = try_UMN(2,params_extraction, params_training, verbose = 3)
acc, auc, best_params = try_CVD(params_extraction, params_training, verbose = 2)


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
# {"L":5, "t1":-5, "t2":1, "min_motion":0.05, "fast_threshold":10, "others":{}} 
# C: 32, bins = 32
# AUC: 0.86

# {'n_bins': 32, 'hidden_layer_sizes': (64, 8), 'solver': 'adam', 'alpha': 0.001}
# Accuracy: 0.875
# AUC: 0.877


