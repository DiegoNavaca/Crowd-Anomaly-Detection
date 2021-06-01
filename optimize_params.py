from main import try_CVD
from main import try_UMN
import optuna

def objective(trial):
    #C = trial.suggest_int('C',2,512)
    bins = trial.suggest_int('bins',16,512)
    autoencoder = trial.suggest_int('autoencoder',16,512)
    au_dropout = trial.suggest_float("dropout",0.0,0.5)
    class_loss = trial.suggest_categorical('class_loss',["binary_crossentropy","poisson","kl_divergence"])
    encoder_layers = trial.suggest_int('extra_encoder_layers',0,4)
    decoder_layers = trial.suggest_int('extra_decoder_layers',0,4)
    class_layers = trial.suggest_int('extra_class_layers',0,4)
    n_parts = trial.suggest_int('n_video_parts',1,4)

    params_extraction = {"L":10, "t1":-5, "t2":1, "min_motion":0.025, "fast_threshold":10, "others":{}}
    params_autoencoder = {"activation":'relu', "batch_norm": True, "extra_encoder_layers": encoder_layers, "extra_decoder_layers": decoder_layers, "extra_class_layers": class_layers,
                          "dropout":au_dropout, "class_loss":class_loss, "classifier_act":"softmax"}
    params_training = {"C":[1,2,4,8,16,32,64,128,150,200,250,300,350,400,450,500,550,600]}
    
    params = {"training":params_training,"extraction":params_extraction,
              "autoencoder":params_autoencoder, "bins":[bins], "code_size":[autoencoder],
              "n_parts":n_parts, 'eliminar_descriptores':[]}
    
    _, auc, _ = try_CVD(params, verbose = 0)

    return auc

study = optuna.create_study(direction='maximize', study_name = 'CVD_AU', storage="sqlite:///Optuna Databases/CVD_AU.db", load_if_exists = True)
#study.optimize(objective, n_trials = 200)

print(study.best_params, study.best_value)
print()
print(optuna.importance.get_param_importances(study))
