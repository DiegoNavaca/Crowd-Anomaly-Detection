import glob
import os
import numpy as np
import keras

from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from sklearn.model_selection import KFold

from data import prepare_Hist_and_Labels
from data import get_Range_Descriptors

class Autoencoder(Model):
    def __init__(self, input_size, latent_dim):
        act = 'relu'
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            Dense(input_size // 2, activation=act),
            Dense(input_size // 4, activation=act),
            Dropout(1/(latent_dim+1)),
            Dense(latent_dim, activation=act),
        ])
        self.decoder = keras.Sequential([
            Dense(input_size // 4, activation=act),
            Dense(input_size // 2, activation=act),
            Dense(input_size, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def cross_validate_encoder(x_train, model, params, verbose = 0):
    result = 0
    n_folds = 5
    for train_index, test_index in KFold(n_folds).split(X = x_train):
        train, test = x_train[train_index], x_train[test_index]
        model.fit(train, train, verbose = verbose, **params)
        result += model.evaluate(test, test, verbose = 0)

    result /= n_folds
    
    return result
    
def prepare_encoder(descriptors_dir, bin_vals, params, loss, is_video_classification, eliminar_descriptores = [], exclude_anomalies = False, verbose = 1):
    output_dir = "Encoders/"+descriptors_dir.split('/', 1)[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for n_bins in bin_vals:
        names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]
        
        range_max, range_min = get_Range_Descriptors(names, is_video_classification)
        
        hist, labels = prepare_Hist_and_Labels(names,range_max, range_min, is_video_classification, n_bins, eliminar_descriptores, eliminar_vacios = False)

        print(len(hist),len(labels))
        if exclude_anomalies:
            hist = [hist[i] for i in range(len(hist)) if labels[i] != -1]

        hist = np.array(hist)
    
        model = Autoencoder(len(hist[0]),params.pop("code_size"))
        model.compile(optimizer='adam', loss=loss)
        
        model.fit(hist,hist, verbose = verbose ,**params)
            
        model.encoder.save(output_dir+"encoder"+str(n_bins)+".h5")

########################################################################################
        
import matplotlib.pyplot as plt
from keras.models import load_model
def visualice_data(descriptors_dir,n_bins, is_video_classification):
    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]
        
    range_max, range_min = get_Range_Descriptors(names, is_video_classification)
        
    hist, labels = prepare_Hist_and_Labels(names,range_max, range_min, is_video_classification, n_bins, [], eliminar_vacios = True)

    encoder_dir = "Encoders/"+descriptors_dir.split("/",1)[1]
    encoder_file = encoder_dir+"encoder"+str(n_bins)+".h5"
    encoder = load_model(encoder_file, compile=False)
    hist = encoder.predict(np.array(hist))

    if len(hist[0]) > 2:
        ax = plt.axes(projection ="3d")
    for label in [-1,1]:
        x = [hist[i][0] for i in range(len(hist)) if labels[i] == label]
        y = [hist[i][1] for i in range(len(hist)) if labels[i] == label]
        if len(hist[0]) > 2:
            z = [hist[i][2] for i in range(len(hist)) if labels[i] == label]
            ax.scatter3D (x,y,z, label = label)
        else:
            plt.scatter(x,y, label = label)
    plt.legend()
    plt.show()

########################################################################################

params = {"code_size":16,"epochs":25}
loss = 'mean_squared_error'
bin_vals = [64]
#descriptors_dir = "Descriptors/UMN/Escena 1/"
descriptors_dir = "Descriptors/CVD/"
if params["code_size"] < 4:
    prepare_encoder(descriptors_dir,bin_vals, params, loss,
                False, exclude_anomalies = False)
else:
    prepare_encoder(descriptors_dir,bin_vals, params, loss,
                False, exclude_anomalies = False)
    for b in bin_vals:
        visualice_data(descriptors_dir,b, True)
