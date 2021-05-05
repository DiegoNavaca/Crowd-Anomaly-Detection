import glob
import time
import numpy as np

from math import ceil
from operator import mul
from functools import reduce

from files import extract_Descriptors_Dir
from files import get_Ground_Truth
from models import train_and_Test

def try_Dataset(dataset, descriptors_dir, video_dir, params, tam_test, verbose = 2, is_video_classification = False, skip_extraction = True):
    np.random.seed(5)
    
    gt = get_Ground_Truth("Datasets/"+dataset+"/ground_truth.txt")

    start = time.time()
    extract_Descriptors_Dir(params["extraction"], video_dir, descriptors_dir,
                            gt, verbose, is_video_classification, skip_extraction)
    if verbose > 0 and not skip_extraction:
        print("Tiempo de extraccion total: {:1.3f}".format(time.time()-start))

    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]
    if "OC" in params["training"]:
        if is_video_classification:
            normal_videos = [name for name in names if gt[name.split("/")[-1]] == 1]
            anomalies = [name for name in names if gt[name.split("/")[-1]] == -1]
        else:
            print("Funcionalidad no implementada")
            return 0,0,{}
    else:
        np.random.shuffle(names)
        
    n_folds = ceil(len(names)/tam_test)
    n_pruebas = reduce(mul, [len(params["training"][x]) for x in params["training"]], 1)*len(params["bins"])*len(params["code_size"])    
    
    average_acc = np.zeros(n_pruebas)
    average_auc = np.zeros(n_pruebas)
    for i in range(n_folds):
        start = time.time()
        if "OC" in params["training"]:
            np.random.shuffle(normal_videos)
            training = normal_videos[:(len(normal_videos)//2)+1]
            test = anomalies + normal_videos[(len(normal_videos)//2)+1:]
        else:
            test = names[i*tam_test:i*tam_test+tam_test]
            training = names[:i*tam_test]+names[i*tam_test+tam_test:]

        acc, auc, list_params = train_and_Test(training, test, is_video_classification, params, verbose = verbose-1)
        
        if verbose > 0:
            print("Time:",time.time()-start)
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

#######################################################################

def try_UMN(escena, params, verbose = 2, skip_extraction = True):
    descriptors_dir = "Descriptors/UMN/Escena "+str(escena)+"/"
    video_dir = "Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/"

    acc, auc, best_params = try_Dataset("UMN", descriptors_dir, video_dir, params,
                                        1, verbose-1, skip_extraction = skip_extraction)

    if verbose > 0:
        print("RESULTADOS:")
        print(best_params)
        print("Accuracy: {:1.3f}".format(acc))
        print("AUC: {:1.3f}".format(auc))

    return acc, auc, best_params

def try_CVD(params, verbose = 2, skip_extraction = True):
    descriptors_dir = "Descriptors/CVD/"
    video_dir = "Datasets/Crowd Violence Detection/"

    #params["use_sift"] = 2000

    acc, auc, best_params = try_Dataset("Crowd Violence Detection", descriptors_dir, video_dir, params, 50, verbose-1, is_video_classification = True, skip_extraction = skip_extraction)

    if verbose > 0:
        print("RESULTADOS:")
        print(best_params)
        print("Accuracy: {:1.3f}".format(acc))
        print("AUC: {:1.3f}".format(auc))
    
    return acc, auc, best_params
    
######################################################################

# def try_CUHK(params, verbose = 2):
#     descriptors_dir = "Descriptors/CUHK/"
#     video_dir = "Datasets/CUHK/Videos/"

#     #params["others"]["skip_to_middle"] = True
#     params["others"]["skip_frames"] = 1
#     params["others"]["num_frames"] = 1
#     params["others"]["change_resolution"] = 320

#     acc, conf_mat, C, n_bins = try_Multiclass("CUHK", descriptors_dir, video_dir, params, 25, verbose, remove_same_scenes = False, C_vals = (8,16,32,64,128))

#     if verbose > 0
#         print("RESULTADOS:")
#         print("C: {}\tNº bins: {}".format(C,n_bins))
#         print("Accuracy: {:1.3f}".format(acc))
#         print("{}".format(conf_mat)) # row: class   column: prediction
    
#     return acc, conf_mat, C, n_bins

############################################################################

############################################################################

if __name__ == "__main__":
    params_extraction = {'L': 20, 't1': -3, 't2': 2, 'min_motion': 0.05,
                         "fast_threshold":20, "others":{}}
    params_autoencoder = {"activation":"relu","dropout":0.3,"batch_norm":True,
                          'extra_class_layers': 0, 'extra_encoder_layers': 1,
                          'extra_decoder_layers': 1, "class_loss":"kl_divergence",
                          "classifier_act":"softmax"}
    params_training = {"C":[1,4,16,64]}
    params = {"extraction":params_extraction, "autoencoder":params_autoencoder,
              "training":params_training, "bins":[200],
              "code_size":[16], "n_parts":1}    

    acc, auc, best_params = try_UMN(3,params, verbose = 3, skip_extraction = False)
    #acc, auc, best_params = try_CVD(params, verbose = 3)


################################ Resultados ################################

# Escena 1
# {'L': 15, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# {'n_bins': 16, 'code_size': None, 'C': 4}
# Accuracy: 0.99
# AUC: 0.987
# {'n_bins': 64, 'code_size': 0.95, 'C': 4}
# Accuracy: 0.988
# AUC: 0.971
# {'n_bins': 200, 'code_size': 64, 'C': 64}
# Accuracy: 0.979
# AUC: 0.967

# Nº frames: 622, 825
# Umbral, ACC, AUC, T. medio, fps
# 5, 0.983, 0.959, 380.7, 3.8
# 10, 0.990, 0.981, 262.9, 5.5
# 15, 0.991, 0.985, 168.4, 8.5
# 20, 0.988, 0.981, 121.9, 11.8
# 25, 0.991, 0.984, 96.0, 15.0
# 30, 0.989, 0.979, 83.1, 17.4
# 35, 0.988, 0.974, 74.2, 19.5
# 40, 0.988, 0.965, 61.5, 23.5

# Con blur (3,3)
# Umbral, ACC, AUC, T. medio
# 5, 0.992, 0.985, 192.3
# 10, 0.989, 0.972, 118.3
# 15, 0.986, 0.969, 81.1
# 20, 0.982, 0.958, 64.2
# 25, 0.977, 0.942, 47.2

# Escena 2
# {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
# {'n_bins': 64, 'code_size': None, 'C': 1}
# Accuracy: 0.947
# AUC: 0.918
# {'n_bins': 128, 'code_size': 0.975, 'C': 1}
# Accuracy: 0.949
# AUC: 0.921
# {'n_bins': 200, 'code_size': 128, 'C': 1}
# Accuracy: 0.944
# AUC: 0.922

# Nº frames: 546, 681, 765, 576, 891, 666 = 4125
# Umbral, ACC, AUC, T. medio, fps
# 45, 0.938, 0.902, 121.892, 33.8
# 40, 0.938, 0.914, 127.6, 32.3
# 35, 0.940, 0.915, 142.8, 28.8
# 30, 0.933, 0.905, 166.2, 24.8
# 25, 0.930, 0.904, 197.5
# 20, 0.940, 0.905, 242.7
# 15, 0.930,0.893, 311.8
# 10, 0.947, 0.918, 428.7

# Escena 3
# {'L': 20, 't1': -3, 't2': 2, 'min_motion': 0.05, 'fast_threshold': 20}
# {'n_bins': 32, 'code_size': None, 'C': 1}
# Accuracy: 0.987
# AUC: 0.973
# {'n_bins': 100, 'code_size': 0.95, 'C': 1} 
# Accuracy: 0.986
# AUC: 0.972
# {'n_bins': 200, 'code_size': 16, 'C': 1}
# Accuracy: 0.984
# AUC: 0.984

# Nº frames: 654, 672, 804 = 2130
# Umbral, ACC, AUC, T. medio, fps
# 5, 0.984, 0.970, 612.8, 3.4
# 10, 0.984, 0.971, 421.3, 5.1
# 15, 0.986, 0.971, 325.7, 6.5
# 20, 0.986, 0.972, 266.7, 8.0
# 25, 0.983, 0.970, 228.8, 9.3
# 30, 0.989, 0.972, 191,7, 11.1
# 35, 0.987, 0.971, 166,7, 12.7
# 40, 0.990, 0.969, 140.6, 15.1
# 45, 0.986, 0.943, 129.4, 16.4
# 50, 0.984, 0.936, 107.7, 19.7

# CVD
# {"L":5, "t1":-5, "t2":1, "min_motion":0.05, "fast_threshold":10, "others":{}} 
# {'n_bins': 128, 'code_size': 0.95}
# Accuracy: 0.890
# AUC: 0.890

# code_size: None
# Accuracy: 0.862
# AUC: 0.864

# code_size: 250 bins: 150
# Accuracy: 0.870
# AUC: 0.870

