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
    params_extraction = {"L":5, "t1":-5, "t2":1, "min_motion":0.025,
                         "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":"relu","dropout":0.3,"batch_norm":True,
                          'extra_class_layers': 1, 'extra_encoder_layers': 1,
                          'extra_decoder_layers': 1, "class_loss":"binary_crossentropy",
                          "classifier_act":"softmax"}
    params_training = {"C":[1,2,4,8,16,32,64,128,256]}
    params = {"extraction":params_extraction, "autoencoder":params_autoencoder,
              "training":params_training, "bins":[200],"code_size":[0.95],
              "n_parts":1}
    

    #acc, auc, best_params = try_UMN(2,params, verbose = 3)
    acc, auc, best_params = try_CVD(params, verbose = 3)


################################ Resultados ################################

# Escena 1
# {'L': 15, 't1': -5, 't2': 1, 'min_motion': 0.025, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.992
# AUC: 0.990

# Escena 2
# {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
# {'n_bins': 64, 'C': 64}
# Accuracy: 0.954
# AUC: 0.927

# {'n_bins': 64, 'hidden_layer_sizes': (128, 32, 4), 'solver': 'adam', 'alpha': 0.0001}
# Accuracy: 0.959
# AUC: 0.935

# Escena 3
# {'L': 20, 't1': -3, 't2': 2, 'min_motion': 0.05, 'fast_threshold': 20}
# nu: 0.01
# Accuracy: 0.983
# AUC: 0.973

# CVD
# {"L":5, "t1":-5, "t2":1, "min_motion":0.05, "fast_threshold":10, "others":{}} 
# {'n_bins': 128, 'code_size': 0.95}
# Accuracy: 0.890
# AUC: 0.890

# code_size: None
# Accuracy: 0.862
# AUC: 0.864

# code_size: 250 bins: 150
# Accuracy: 0.874
# AUC: 0.873
