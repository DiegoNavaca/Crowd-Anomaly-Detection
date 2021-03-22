import glob
import os
import numpy as np
import keras

from keras.layers import Dense
from keras.models import Model
from sklearn.model_selection import KFold

from data import prepare_Hist_and_Labels
from data import get_Range_Descriptors

class Autoencoder(Model):
    def __init__(self, input_size, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            Dense(input_size // 2, activation='relu'),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = keras.Sequential([
            Dense(input_size, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_model(input_size = 1024):
    hidden_size = input_size // 2
    code_size = 8
    input_img = keras.models.Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_img)
    code = Dense(code_size, activation='relu')(hidden_1)
    #ar = keras.regularizers.l1(10e-5)(input_img)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_img = Dense(input_size, activation='sigmoid')(hidden_2)
    autoencoder = keras.models.Model(input_img, output_img)
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')#'binary_crossentropy')

    return autoencoder

def train_encoder(x_train, model, params, verbose = 0):
    result = 0
    n_folds = 5
    for train_index, test_index in KFold(n_folds).split(X = x_train):
        train, test = x_train[train_index], x_train[test_index]
        model.fit(train, train, verbose = verbose, **params)
        result += model.evaluate(test, test, verbose = 0)

    result /= n_folds
    
    return result
    
def prepare_encoder(descriptors_dir, bin_vals, params, eliminar_descriptores = [], verbose = 1):
    output_dir = "Encoders/"+descriptors_dir.split('/', 1)[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for n_bins in bin_vals:
        names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]
        
        range_max, range_min = get_Range_Descriptors(names, is_video_classification = False)
        
        hist, _ = prepare_Hist_and_Labels(names,range_max, range_min, False, n_bins,
                                      eliminar_descriptores, eliminar_vacios = False)

        hist = np.array(hist)
    
        model = Autoencoder(len(hist[0]),8)
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        result = train_encoder(hist, model, params, verbose = verbose-1)

        if verbose > 0:
            print("\nValidation loss: {}".format(result))
        
        model.fit(hist,hist, verbose = verbose ,**params)
            
        model.encoder.save(output_dir+"encoder"+str(n_bins)+".h5")

params = {"epochs":3}
#prepare_encoder("Descriptors/UMN/Escena 1/",[64,128,256], params)
prepare_encoder("Descriptors/CVD/",[64,128,256], params)
