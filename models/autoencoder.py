import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class Autoencoder(models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        # predict() loops over the data in batches
        # it's equivalent to
        # y_batch = []
        # for x in x_batch:
        #   y = model(x).numpy()
        #   y_batch.append(y)
        encoded = self.encoder(x)
        decoded = self.encoder(encoded)

        return decoded

class Encoder(models.Model):
    def __init__(self):
        super(Encoder, self).__init__()

class Decoder(models.Model):
    def __init__(self):
        super(Decoder, self).__init__()


# resnet50: https://colab.research.google.com/drive/1IWWqlc0KhJ7JaAF1Subu2DYAICwRN1_n
# https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py
# https://www.tensorflow.org/tutorials/generative/autoencoder#define_a_convolutional_autoencoder