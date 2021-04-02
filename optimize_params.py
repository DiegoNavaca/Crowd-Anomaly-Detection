from main import try_CVD
from main import try_UMN
import optuna

def objective(trial):
    C = trial.suggest_int('C',2,128)
    bins = trial.suggest_int('bins',16,256)
    autoencoder = trial.suggest_int('autoencoder',8,128)
    au_activation = trial.suggest_categorical("AU activation", ['relu','linear'])
    au_extra_layers = trial.suggest_int("extra_layers",0,4)
    au_batch_norm = trial.suggest_categorical("BatchNormalization", [True,False])
    au_dropout = trial.suggest_float("dropout",0.0,0.5)

    params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":au_activation, "batch_norm": au_batch_norm, "extra_layers": au_extra_layers, "dropout":au_dropout}
    params_training = {"C":[C], "bins":[bins], "code_size":[autoencoder], "params_autoencoder":params_autoencoder}
    _, auc, _ = try_CVD(params_extraction, params_training, verbose = 1)

    return auc

study = optuna.create_study(direction='maximize', study_name = 'CVD', storage="sqlite:///CVD.db", load_if_exists = True)
study.optimize(objective, n_trials = 10)

print(study.best_params)
