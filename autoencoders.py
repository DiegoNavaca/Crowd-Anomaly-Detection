import glob
import numpy as np
import keras

from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_size, latent_dim, params):
        act = params['activation']
        super().__init__()
        self.latent_dim = latent_dim
        layers = [Dense(latent_dim, activation=act)]
        if 'dropout' in params:
            layers.insert(0,Dropout(params['dropout']))
        if params.get('batch_norm',False):
            layers.insert(-1,BatchNormalization())
        if 'extra_coder_layers' in params:
            for i in range(params['extra_decoder_layers']):
                layers.insert(0,Dense(latent_dim+input_size // 2**(i+1)))
        
        self.encoder = keras.Sequential(layers, name = "encoder")
        input_layer = keras.Input(latent_dim)
        
        if params.get('extra_decoder_layers',0) > 0:
            x = Dense(latent_dim+input_size // 2**params['extra_decoder_layers'], activation=act)(input_layer)
            for i in range(params['extra_decoder_layers']-2,0,-1):
                x = Dense(latent_dim+input_size // 2**(i+1))(x)
            self.decoder = Dense(input_size, activation='sigmoid', name = "decoder")(x)
        else:
            self.decoder = Dense(input_size, activation='sigmoid', name = "decoder")(input_layer)

        if params.get('extra_class_layers',0) > 0:
            x = Dense(latent_dim+input_size // 2, activation=act)(input_layer)
            for i in range(1,params['extra_decoder_layers']):
                x = Dense(latent_dim // 2**(i+1))(x)
            self.classifier_layer = Dense(2, activation = params["classifier_act"], name = "classifier")(x)
        else:
            self.classifier_layer = Dense(2, activation = params["classifier_act"], name = "classifier")(input_layer)
        
        self.salida = keras.Model(inputs = input_layer, outputs = [self.classifier_layer, self.decoder])
        self.classifier_model = keras.Model(inputs = input_layer, outputs = self.classifier_layer)

    def call(self, x):
        encoded = self.encoder(x)
        return self.salida(encoded)
        

################################# VISUALIZATION ##########################################

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import losses
from keras.metrics import BinaryAccuracy
from data import get_Range_Descriptors
from data import prepare_Hist_and_Labels

def visualice_data(descriptors_dir,n_bins, is_video_classification):
    
    names = [name[:-5] for name in glob.glob(descriptors_dir+"*.data")]
        
    range_max, range_min = get_Range_Descriptors(names, is_video_classification)
        
    hist, labels = prepare_Hist_and_Labels(names,range_max, range_min, is_video_classification, n_bins, [], eliminar_vacios = True)

    params = {'activation':'relu', 'extra_layers':1,'dropout':0.4,'batch_norm':True}
    autoencoder = Autoencoder(len(hist[0]),2,params)

    autoencoder.compile(optimizer = 'adam',
                                    loss = {"output_2":losses.MeanSquaredError(),
                                            "output_1":losses.Poisson()},
                                    metrics = {"output_1":BinaryAccuracy(name='acc')})
    autoencoder.fit(hist,{"output_2":hist,"output_1":labels},
                                          verbose = 1, epochs = 20,
                                          validation_split = 0.2)
    hist = autoencoder.encoder.predict(np.array(hist))

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

if __name__ == "__main__":
    visualice_data("Descriptors/CVD/",32,True)
