from main import try_CVD
from main import try_UMN
import optuna

def objective(trial):
    C = trial.suggest_int('C',2,256)
    bins = trial.suggest_int('bins',16,512)
    autoencoder = trial.suggest_int('autoencoder',8,256)
    au_dropout = trial.suggest_float("dropout",0.0,0.5)
    class_loss = trial.suggest_categorical('class_loss',["binary_crossentropy","poisson","kldivergence"])

    params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":'relu', "batch_norm": True, "extra_layers": 1, "dropout":au_dropout, "class_loss":class_loss}
    params_training = {"C":[C], "bins":[bins], "code_size":[autoencoder], "params_autoencoder":params_autoencoder}
    _, auc, _ = try_CVD(params_extraction, params_training, verbose = 1)

    return auc

study = optuna.create_study(direction='maximize', study_name = 'CVD_AU', storage="sqlite:///CVD_AU.db", load_if_exists = True)
study.optimize(objective, n_trials = 10)

#study = optuna.load_study(study_name = 'CVD_AU', storage = 'sqlite:///Optuna Databases/CVD_AU.db')
print(study.best_params, study.best_value)
print(optuna.importance.get_param_importances(study))
