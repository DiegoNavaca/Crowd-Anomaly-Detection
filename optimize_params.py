from main import try_CVD
from main import try_UMN
import optuna

def objective(trial):
    C = trial.suggest_int('C',2,256)
    bins = trial.suggest_int('bins',32,512)
    autoencoder = trial.suggest_int('autoencoder',8,256)
    au_dropout = trial.suggest_float("dropout",0.0,0.5)
    classifier_act = trial.suggest_categorical("class_act",["sigmoid","softmax"])
    class_loss = trial.suggest_categorical('class_loss',["binary_crossentropy","poisson","kl_divergence"])
    n_parts = trial.suggest_int('n_video_parts',1,5)

    params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":'relu', "batch_norm": True, "extra_layers": 1, "dropout":au_dropout, "class_loss":class_loss, "classifier_act":classifier_act}
    params_training = {"C":[C]}
    
    params = {"training":params_training,"extraction":params_extraction,
              "autoencoder":params_autoencoder, "bins":[bins], "code_size":[autoencoder],
              "n_parts":n_parts}
    
    _, auc, _ = try_CVD(params, verbose = 1)

    return auc

study = optuna.create_study(direction='maximize', study_name = 'CVD_AU_DIV', storage="sqlite:///CVD_AU_DIV.db", load_if_exists = True)
study.optimize(objective, n_trials = 200)

#study = optuna.load_study(study_name = 'CVD', storage = 'sqlite:///Optuna Databases/CVD.db')
print(study.best_params, study.best_value)
print(optuna.importance.get_param_importances(study))
