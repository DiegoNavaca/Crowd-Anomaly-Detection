from main import try_CVD
from main import try_UMN
import optuna

def objective(trial):
    C = trial.suggest_int('C',2,256)
    bins = trial.suggest_int('bins',32,512)
    autoencoder = trial.suggest_int('autoencoder',16,256)
    au_epochs = trial.suggest_int('epochs',10,50)
    au_dropout = trial.suggest_float("dropout",0.0,0.5)
    classifier_act = trial.suggest_categorical("class_act",["sigmoid","softmax"])
    class_loss = trial.suggest_categorical('class_loss',["binary_crossentropy","poisson","kl_divergence"])
    coder_layers = trial.suggest_int('extra_coder_layers',0,4)
    decoder_layers = trial.suggest_int('extra_decoder_layers',0,4)
    class_layers = trial.suggest_int('extra_class_layers',0,4)
    n_parts = trial.suggest_int('n_video_parts',1,5)
    remove_descriptor = trial.suggest_int('remove_descriptor',0,10)

    params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":'relu', "batch_norm": True, "extra_coder_layers": coder_layers, "extra_decoder_layers": decoder_layers, "extra_class_layers": class_layers,
                          "dropout":au_dropout, "class_loss":class_loss, "classifier_act":classifier_act, "epochs":au_epochs}
    params_training = {"C":[C]}
    
    params = {"training":params_training,"extraction":params_extraction,
              "autoencoder":params_autoencoder, "bins":[bins], "code_size":[autoencoder],
              "n_parts":n_parts, 'eliminar_descriptores':[remove_descriptor]}
    
    _, auc, _ = try_CVD(params, verbose = 0)

    return auc

#study = optuna.create_study(direction='maximize', study_name = 'CVD_AU', storage="sqlite:///CVD_AU.db", load_if_exists = True)
#study.optimize(objective, n_trials = 200)

study = optuna.load_study(study_name = 'CVD_AU', storage = 'sqlite:///Optuna Databases/CVD_AU.db')
print(study.best_params, study.best_value)
print()
print(optuna.importance.get_param_importances(study,evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()))
