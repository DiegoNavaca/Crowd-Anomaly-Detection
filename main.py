import glob
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from files import extract_Descriptors_Dir
from files import get_Ground_Truth
from SVM import get_Max_Descriptors
from SVM import train_SVC
from SVM import test_SVM
from SVM import prepare_Hist_and_Labels

def try_UMN(escena, params, tam_test, verbose = True):

    descriptors_dir = "Descriptors/UMN/Escena "+str(escena)+"/"
    
    gt = get_Ground_Truth("Datasets/UMN/ground_truth.txt")
    
    extract_Descriptors_Dir(params, "Datasets/UMN/Escenas Completas/Escena "+str(escena)+"/", descriptors_dir,gt)

    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]


    best_acc = 0
    best_auc = 0
    best_nu = 0.1
    for nu in (0.1, 0.01, 0.001):
        average_acc = 0
        average_auc = 0
        for i in range(len(names)):
            test = names[i*tam_test:]+names[:i*tam_test+tam_test]
            training = names[:i*tam_test]+names[i*tam_test+tam_test:]
            
            range_max = get_Max_Descriptors(training)

            hist, labels = prepare_Hist_and_Labels(training, range_max)

            model = train_SVC(hist, labels, nu = nu)

            hist, labels = prepare_Hist_and_Labels(test, range_max)

            prediction = test_SVM(hist, range_max, model)

            acc = accuracy_score(labels,prediction)
            auc = roc_auc_score(labels,prediction)

            if verbose:
                print("Nombre: {}".format(names[i]))
                print("Acertados: ",sum(1 for i in range(len(prediction)) if prediction[i] == labels[i]),"-",len(prediction))
                print("Positivos: {} - {}".format(
                    sum(1 for i in range(len(prediction)) if prediction[i] == labels[i] and prediction[i] == -1), labels.count(-1)))
                print([(i+params["L"]-1, labels[i]) for i in range(len(prediction)) if prediction[i] != labels[i]])
                print("Accuracy: {:1.3f}".format(acc))
                print("AUC: {:1.3f}\n".format(auc))

            average_acc += acc
            average_auc += auc

        average_acc /= len(names)
        average_auc /= len(names)

        if verbose:
            print("nu: {}".format(nu))
            print("Accuracy: {:1.3f}".format(average_acc))
            print("AUC: {:1.3f}\n".format(average_auc))

        if average_auc > best_auc:
            best_acc = average_acc
            best_auc = average_auc
            best_nu = nu

    return best_acc, best_auc, best_nu
        

for L in (5,10,15,20):
    for t1 in (-3,-5):
        for t2 in (1,2,4):
            for min_motion in (0.01,0.025, 0.05):
                params = {"L":L, "t1":t1, "t2":t2, "min_motion":min_motion, "fast_threshold":20}
                print(params)
                acc, auc, nu = try_UMN(3, params, 1, verbose = False)

                print("nu: {}".format(nu))
                print("Accuracy: {:1.3f}".format(acc))
                print("AUC: {:1.3f}".format(auc))

# params = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":20}
# print(params)
# acc, auc, nu = try_UMN(2,params, verbose = True)

# print("nu: {}".format(nu))
# print("Accuracy: {:1.3f}".format(acc))
# print("AUC: {:1.3f}".format(auc))

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
