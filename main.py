import glob
import time
import numpy as np

from math import ceil
from operator import mul
from functools import reduce
from json import load

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

    if params["extraction"] is None:
        conf = open("config.json")
        params["extraction"] = load(conf)["UMN"+str(escena)]
        conf.close()

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

    if params["extraction"] is None:
        conf = open("config.json")
        params["extraction"] = load(conf)["CVD"]
        conf.close()

    acc, auc, best_params = try_Dataset("Crowd Violence Detection", descriptors_dir, video_dir, params, 50, verbose-1, is_video_classification = True, skip_extraction = skip_extraction)

    if verbose > 0:
        print("RESULTADOS:")
        print(best_params)
        print("Accuracy: {:1.3f}".format(acc))
        print("AUC: {:1.3f}".format(auc))
    
    return acc, auc, best_params

############################################################################

############################################################################

if __name__ == "__main__":
    params_extraction = None
    params_autoencoder = {"activation":"relu","dropout":0.3,"batch_norm":True,
                    'extra_class_layers': 0, 'extra_encoder_layers': 1,
                    'extra_decoder_layers': 1, "class_loss":"kl_divergence",
                    "classifier_act":"softmax"}
    params_training = {"C":[1,4,16,32,64,128]}
    params = {"extraction":params_extraction, "autoencoder":params_autoencoder,
              "training":params_training, "bins":[16],
              "code_size":[None], "n_parts":1}    

    acc, auc, best_params = try_UMN(3,params, verbose = 2, skip_extraction = False)
    #acc, auc, best_params = try_CVD(params, verbose = 3, skip_extraction = True)


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

# Nº frames: 622, 825 = 1447
# Umbral, ACC, AUC, Tiempo, fps
# 5, 0.982, 0.962, 318.0, 4.5
# 10, 0.987, 0.974, 220.1, 6.5
# 15, 0.994, 0.989, 139.7, 10.3
# 20, 0.992, 0.984, 101.6, 14.2
# 25, 0.991, 0.983, 79.5, 18.2
# 30, 0.987, 0.974, 68.8, 21.0
# 35, 0.988, 0.972, 59.7, 24.2
# 40, 0.978, 0.949, 48.8, 29.65

# Escena 2
# {"L":10, "t1":-5, "t2":1, "min_motion":0.01, "fast_threshold":10, "others":{}}
# {'n_bins': 64, 'code_size': None, 'C': 1}
# Accuracy: 0.961
# AUC: 0.943
# {'n_bins': 128, 'code_size': 0.95, 'C': 1}
# Accuracy: 0.952
# AUC: 0.944
# {'n_bins': 200, 'code_size': 128, 'C': 1}
# Accuracy: 0.957
# AUC: 0.946

# Nº frames: 546, 681, 765, 576, 891, 666 = 4125
# Umbral, ACC, AUC, Tiempo, fps
# 1, 0.939, 0.931, 852.1, 4.8
# 2, 0.959, 0.947, 771.4, 5.3
# 5, 0.949, 0.929, 553.15, 7.5
# 10, 0.943, 0.913, 338.848, 12.1
# 15, 0.933, 0.897, 252.0, 16.3
# 20, 0.936, 0.905, 187.7, 22
# 25, 0.933, 0.901, 153.4, 26.9
# 30, 0.936, 0.907, 129.2, 32
# 35, 0.938,0.912, 110.7, 37.2
# 40, 0.936, 0.905, 96, 43

# Escena 3
# {'L': 20, 't1': -5, 't2': 1, 'min_motion': 0.05, 'fast_threshold': 20}
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
# Umbral, ACC, AUC, Tiempo, fps
# 5, 0.984, 0.971, 453.3, 4.7
# 10, 0.985, 0.971, 312.5, 6.8
# 15, 0.986, 0.971, 235.7, 9.0
# 20, 0.987, 0.973, 195.0, 11.2
# 25, 0.983, 0.970, 164.2, 13.0
# 30, 0.989, 0.972, 140.1, 15.2
# 35, 0.987, 0.971, 120.4, 17.7
# 40, 0.987, 0.966, 103.7, 20.5
# 45, 0.985, 0.943, 94.8, 22.5
# 50, 0.977, 0.930, 84, 25.4

# CVD
# {"L":5, "t1":-5, "t2":1, "min_motion":0.05, "fast_threshold":10, "others":{}} 
# {code_size: None, n_bins = 64}
# Accuracy: 0.862
# AUC: 0.867

# {'n_bins': 128, 'code_size': 0.95}
# Accuracy: 0.894
# AUC: 0.898

# code_size: 250 bins: 150
# Accuracy: 0.870
# AUC: 0.870

# Nº frames: 3.6*30*246 = 26568
# Umbral, ACC, AUC, Tiempo, fps
# 10, 0.862, 0.867, 3810.5, 7
