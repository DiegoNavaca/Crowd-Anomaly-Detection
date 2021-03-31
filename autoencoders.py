import glob
import numpy as np
import keras

from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_size, latent_dim):
        act = 'relu'
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            Dense(latent_dim+input_size // 2, activation=act),
            #Dense(latent_dim+input_size // 4, activation=act),
            Dense(latent_dim, activation=act),
            BatchNormalization(),
        ], name = "encoder")
        input_layer = keras.Input(latent_dim)
        
        x = Dense(latent_dim+input_size // 2, activation=act)(input_layer)
        self.decoder = Dense(input_size, activation='sigmoid', name = "decoder")(x)

        x = Dense(latent_dim//2,activation = act,
                  )(input_layer)
        self.classifier = Dense(2, activation = 'softmax', name = "classifier")(x)
        
        self.salida = keras.Model(inputs = input_layer, outputs = [self.classifier, self.decoder])
        self.clasificador = keras.Model(inputs = input_layer, outputs = self.classifier)

    def call(self, x):
        encoded = self.encoder(x)
        return self.salida(encoded)
        

################################# VISUALIZATION ##########################################

import matplotlib.pyplot as plt
from keras.models import load_model
from data import get_Range_Descriptors
from data import prepare_Hist_and_Labels

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
