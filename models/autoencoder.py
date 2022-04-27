import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class Autoencoder(models.Model):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
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
        decoded = self.decoder(encoded)

        return decoded


class Encoder(models.Model):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.sequential = keras.Sequential([
            # layers.ZeroPadding2D(padding=3),

            # 1st stage
            layers.Conv2D(filters=64, kernel_size=7, strides=2),
            layers.BatchNormalization(axis=3),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2),

            # 2nd stage
            ConvolutionalBlock(filters=[64, 64, 256], f=3, s=1),
            IdentityBlock(filters=[64, 64, 256], f=3),
            # IdentityBlock(filters=[64, 64, 256], f=3),

            # 3rd stage
            ConvolutionalBlock(filters=[128, 128, 512], f=3, s=2),
            IdentityBlock(filters=[128, 128, 512], f=3),
            IdentityBlock(filters=[128, 128, 512], f=3),
            # IdentityBlock(filters=[128, 128, 512], f=3),

            # 4th stage
            ConvolutionalBlock(filters=[256, 256, 1024], f=3, s=2),
            IdentityBlock(filters=[256, 256, 1024], f=3),
            IdentityBlock(filters=[256, 256, 1024], f=3),
            # IdentityBlock(filters=[256, 256, 1024], f=3),
            # IdentityBlock(filters=[256, 256, 1024], f=3),
            # IdentityBlock(filters=[256, 256, 1024], f=3),

            # 5th stage
            ConvolutionalBlock(filters=[512, 512, 2048], f=3, s=2),
            IdentityBlock(filters=[512, 512, 2048], f=3),
            # IdentityBlock(filters=[512, 512, 2048], f=3),

            # latent feature
            layers.Conv2D(filters=1, kernel_size=1, strides=1)


        ])

    def call(self, x):
        x = self.sequential(x)
        return x


class Decoder(models.Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.sequential = keras.Sequential([
            layers.Conv2DTranspose(filters=2048, kernel_size=1, strides=1),

            # IdentityBlockTranspose(filters=[512, 512, 2048], f=3),
            IdentityBlockTranspose(filters=[512, 512, 2048], f=3),
            ConvolutionalBlockTranspose(filters=[512, 512, 1024], f=3, s=2),

            # IdentityBlockTranspose(filters=[256, 256, 1024], f=3),
            # IdentityBlockTranspose(filters=[256, 256, 1024], f=3),
            # IdentityBlockTranspose(filters=[256, 256, 1024], f=3),
            IdentityBlockTranspose(filters=[256, 256, 1024], f=3),
            IdentityBlockTranspose(filters=[256, 256, 1024], f=3),
            ConvolutionalBlockTranspose(filters=[256, 256, 512], f=3, s=2),

            # IdentityBlockTranspose(filters=[128, 128, 512], f=3),
            IdentityBlockTranspose(filters=[128, 128, 512], f=3),
            IdentityBlockTranspose(filters=[128, 128, 512], f=3),
            ConvolutionalBlockTranspose(filters=[128, 128, 256], f=3, s=2),

            # IdentityBlockTranspose(filters=[64, 64, 256], f=3),
            IdentityBlockTranspose(filters=[64, 64, 256], f=3),
            ConvolutionalBlockTranspose(filters=[64, 64, 64], f=3, s=1),

            layers.UpSampling2D(size=2),
            layers.ReLU(),
            layers.BatchNormalization(axis=3),
            layers.Conv2DTranspose(filters=3, kernel_size=7, strides=2, activation='sigmoid'),
        ])

    def call(self, x):
        x = self.sequential(x)
        return x

@tf.keras.utils.register_keras_serializable()
class IdentityBlock(layers.Layer):
    def __init__(self, filters, f, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)

        self.filters = filters
        self.f = f

        f1, f2, f3 = filters

        # 1st component
        self.conv1 = layers.Conv2D(filters=f1, kernel_size=1, strides=1, padding='valid')
        self.bn1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.ReLU()

        # 2nd component
        self.conv2 = layers.Conv2D(filters=f2, kernel_size=f, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.ReLU()

        # 3rd component
        self.conv3 = layers.Conv2D(filters=f3, kernel_size=1, strides=1, padding='valid')
        self.bn3 = layers.BatchNormalization(axis=3)

        # shortcut path
        self.add = layers.Add()
        self.relu3 = layers.ReLU()

    def call(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.add([x, x_shortcut])
        x = self.relu3(x)

        return x

    def get_config(self):
        config = super(IdentityBlock, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'f': self.f,

                # 'conv1': self.conv1,
                # 'bn1': self.bn1,
                # 'relu1': self.relu1,
                #
                # 'conv2': self.conv2,
                # 'bn2': self.bn2,
                # 'relu2': self.relu2,
                #
                # 'conv3': self.conv3,
                # 'bn3': self.bn3,
                #
                # 'add': self.add,
                # 'relu3': self.relu3
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class IdentityBlockTranspose(layers.Layer):
    def __init__(self, filters, f, **kwargs):
        super(IdentityBlockTranspose, self).__init__(**kwargs)

        self.filters = filters
        self.f = f

        f1, f2, f3 = filters

        # 1st component
        self.bn1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2DTranspose(filters=f1, kernel_size=1, strides=1, padding='valid')

        # 2nd component
        self.bn2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2DTranspose(filters=f2, kernel_size=f, strides=1, padding='same')

        # 3rd component
        self.bn3 = layers.BatchNormalization(axis=3)
        self.relu3 = layers.ReLU()
        self.conv3 = layers.Conv2DTranspose(filters=f3, kernel_size=1, strides=1, padding='valid')

    def call(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x

    def get_config(self):
        config = super(IdentityBlockTranspose, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'f': self.f,

                # 'conv1': self.conv1,
                # 'bn1': self.bn1,
                # 'relu1': self.relu1,
                #
                # 'conv2': self.conv2,
                # 'bn2': self.bn2,
                # 'relu2': self.relu2,
                #
                # 'conv3': self.conv3,
                # 'bn3': self.bn3,
                # 'relu3': self.relu3,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class ConvolutionalBlock(layers.Layer):
    def __init__(self, filters, f, s, **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)

        self.filters = filters
        self.f = f
        self.s = s

        f1, f2, f3 = filters

        # 1st component
        self.conv1 = layers.Conv2D(filters=f1, kernel_size=1, strides=s, padding='valid')
        self.bn1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.ReLU()

        # 2nd component
        self.conv2 = layers.Conv2D(filters=f2, kernel_size=f, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.ReLU()

        # 3rd component
        self.conv3 = layers.Conv2D(filters=f3, kernel_size=1, strides=1, padding='valid')
        self.bn3 = layers.BatchNormalization(axis=3)

        # shortcut path
        self.conv_short = layers.Conv2D(filters=f3, kernel_size=1, strides=s, padding='valid')
        self.bn_short = layers.BatchNormalization(axis=3)
        self.add = layers.Add()
        self.relu3 = layers.ReLU()

    def call(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x_shortcut = self.conv_short(x_shortcut)
        x_shortcut = self.bn_short(x_shortcut)
        x = self.add([x, x_shortcut])
        x = self.relu3(x)

        return x

    def get_config(self):
        config = super(ConvolutionalBlock, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'f': self.f,
                's': self.s,

                # 'conv1': self.conv1,
                # 'bn1': self.bn1,
                # 'relu1': self.relu1,
                #
                # 'conv2': self.conv2,
                # 'bn2': self.bn2,
                # 'relu2': self.relu2,
                #
                # 'conv3': self.conv3,
                # 'bn3': self.bn3,
                # 'conv_short': self.conv_short,
                # 'bn_short': self.bn_short,
                # 'add': self.add,
                # 'relu3': self.relu3,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class ConvolutionalBlockTranspose(layers.Layer):
    def __init__(self, filters, f, s, **kwargs):
        super(ConvolutionalBlockTranspose, self).__init__(**kwargs)

        self.filters = filters
        self.f = f
        self.s = s

        f1, f2, f3 = filters

        # 1st component
        self.bn1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2DTranspose(filters=f1, kernel_size=1, strides=s, padding='valid')

        # 2nd component
        self.bn2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2DTranspose(filters=f2, kernel_size=f, strides=1, padding='same')

        # 3rd component
        self.bn3 = layers.BatchNormalization(axis=3)
        self.relu3 = layers.ReLU()
        self.conv3 = layers.Conv2DTranspose(filters=f3, kernel_size=1, strides=1, padding='valid')

    def call(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x

    def get_config(self):
        config = super(ConvolutionalBlockTranspose, self).get_config()
        config.update(
            {
                'filters': self.filters,
                'f': self.f,
                's': self.s,

                # 'conv1': self.conv1,
                # 'bn1': self.bn1,
                # 'relu1': self.relu1,
                #
                # 'conv2': self.conv2,
                # 'bn2': self.bn2,
                # 'relu2': self.relu2,
                #
                # 'conv3': self.conv3,
                # 'bn3': self.bn3,
                # 'relu3': self.relu3
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# https://towardsdatascience.com/how-to-write-a-custom-keras-model-so-that-it-can-be-deployed-for-serving-7d81ace4a1f8
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/
# https://www.tensorflow.org/tutorials/customization/custom_layers
# resnet50: https://colab.research.google.com/drive/1IWWqlc0KhJ7JaAF1Subu2DYAICwRN1_n
# https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py
# https://www.tensorflow.org/tutorials/generative/autoencoder#define_a_convolutional_autoencoder
